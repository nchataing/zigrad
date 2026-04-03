const std = @import("std");

pub const Pool = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(*Value),
    param_nodes: std.ArrayList(*Value),

    pub fn init(allocator: std.mem.Allocator) Pool {
        return Pool{
            .allocator = allocator,
            .nodes = .empty,
            .param_nodes = .empty,
        };
    }

    pub fn deinit(self: *Pool) void {
        for (self.nodes.items) |node| {
            self.allocator.destroy(node);
        }
        self.nodes.deinit(self.allocator);
        self.param_nodes.deinit(self.allocator);
    }

    pub fn create(self: *Pool, value: Value) *Value {
        const node = self.allocator.create(Value) catch @panic("OOM");
        node.* = value;
        node.pool = self;
        self.nodes.append(self.allocator, node) catch @panic("OOM");
        if (value.role == .param) {
            self.param_nodes.append(self.allocator, node) catch @panic("OOM");
        }
        return node;
    }

    pub fn params(self: *Pool) []*Value {
        return self.param_nodes.items;
    }
};

pub const Value = struct {
    data: f32,
    grad: f32 = 0,
    role: Role = .internal,
    pool: *Pool = undefined,
    _children: [2]?*Value = .{ null, null },
    _backward: ?*const fn (self: *Value) void = null,

    pub const Role = enum { param, input, internal };

    pub fn param(data: f32, pool: *Pool) *Value {
        return pool.create(.{ .data = data, .role = Role.param });
    }

    pub fn input(data: f32, pool: *Pool) *Value {
        return pool.create(.{ .data = data, .role = Role.input });
    }

    pub fn add(self: *Value, other: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].?.grad += v.grad;
                v._children[1].?.grad += v.grad;
            }
        }.backward;
        return self.pool.create(.{
            .data = self.data + other.data,
            ._children = .{ self, other },
            ._backward = &_backward,
        });
    }

    pub fn sub(self: *Value, other: *Value) *Value {
        return self.add(other.neg());
    }

    pub fn neg(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].?.grad += -v.grad;
            }
        }.backward;
        return self.pool.create(.{
            .data = -self.data,
            ._children = .{ self, null },
            ._backward = &_backward,
        });
    }

    pub fn mul(self: *Value, other: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].?.grad += v.grad * v._children[1].?.data;
                v._children[1].?.grad += v.grad * v._children[0].?.data;
            }
        }.backward;
        return self.pool.create(.{
            .data = self.data * other.data,
            ._children = .{ self, other },
            ._backward = &_backward,
        });
    }

    pub fn pow(self: *Value, exponent: f32) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                const exp = v._children[1].?.data;
                v._children[0].?.grad += v.grad * exp * std.math.pow(f32, v._children[0].?.data, exp - 1);
            }
        }.backward;
        const exp_node = self.pool.create(.{ .data = exponent });
        return self.pool.create(.{
            .data = std.math.pow(f32, self.data, exponent),
            ._children = .{ self, exp_node },
            ._backward = &_backward,
        });
    }

    pub fn tanh(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                const out = v.data;
                v._children[0].?.grad += v.grad * (1 - out * out);
            }
        }.backward;
        return self.pool.create(.{
            .data = std.math.tanh(self.data),
            ._children = .{ self, null },
            ._backward = &_backward,
        });
    }

    pub fn backward(self: *Value) void {
        const allocator = self.pool.allocator;
        var topo: std.ArrayList(*Value) = .empty;
        defer topo.deinit(allocator);
        var visited = std.AutoHashMap(*Value, void).init(allocator);
        defer visited.deinit();

        buildTopo(self, &visited, &topo, allocator);

        self.grad = 1.0;
        var rev = std.mem.reverseIterator(topo.items);
        while (rev.next()) |node| {
            if (node._backward) |backward_fn| {
                backward_fn(node);
            }
        }
    }

    fn buildTopo(v: *Value, visited: *std.AutoHashMap(*Value, void), topo: *std.ArrayList(*Value), allocator: std.mem.Allocator) void {
        if (visited.contains(v)) return;
        visited.put(v, {}) catch return;
        v.grad = 0;
        for (v._children) |maybe_child| {
            if (maybe_child) |child| {
                buildTopo(child, visited, topo, allocator);
            }
        }
        topo.append(allocator, v) catch return;
    }
};

test "add backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(2.0, &pool);
    const b = Value.param(3.0, &pool);
    const c = a.add(b);

    c.backward();

    try std.testing.expectEqual(@as(f32, 5.0), c.data);
    try std.testing.expectEqual(@as(f32, 1.0), a.grad);
    try std.testing.expectEqual(@as(f32, 1.0), b.grad);
}

test "mul backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(2.0, &pool);
    const b = Value.param(3.0, &pool);
    const c = a.mul(b);

    c.backward();

    try std.testing.expectEqual(@as(f32, 6.0), c.data);
    try std.testing.expectEqual(@as(f32, 3.0), a.grad);
    try std.testing.expectEqual(@as(f32, 2.0), b.grad);
}

test "neg backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(3.0, &pool);
    const b = a.neg();

    b.backward();

    try std.testing.expectEqual(@as(f32, -3.0), b.data);
    try std.testing.expectEqual(@as(f32, -1.0), a.grad);
}

test "sub backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(5.0, &pool);
    const b = Value.param(3.0, &pool);
    const c = a.sub(b);

    c.backward();

    try std.testing.expectEqual(@as(f32, 2.0), c.data);
    try std.testing.expectEqual(@as(f32, 1.0), a.grad);
    try std.testing.expectEqual(@as(f32, -1.0), b.grad);
}

test "pow backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(3.0, &pool);
    const b = a.pow(2.0);

    b.backward();

    try std.testing.expectEqual(@as(f32, 9.0), b.data);
    try std.testing.expectEqual(@as(f32, 6.0), a.grad); // 2 * 3^1 = 6
}

test "tanh backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(0.0, &pool);
    const b = a.tanh();

    b.backward();

    try std.testing.expectEqual(@as(f32, 0.0), b.data);
    try std.testing.expectEqual(@as(f32, 1.0), a.grad); // tanh'(0) = 1 - 0^2 = 1
}

test "chain: a*b + c" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(2.0, &pool);
    const b = Value.param(3.0, &pool);
    const c = Value.param(4.0, &pool);
    const d = a.mul(b).add(c);

    d.backward();

    try std.testing.expectEqual(@as(f32, 10.0), d.data);
    try std.testing.expectEqual(@as(f32, 3.0), a.grad);
    try std.testing.expectEqual(@as(f32, 2.0), b.grad);
    try std.testing.expectEqual(@as(f32, 1.0), c.grad);
}

test "params list" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const w = Value.param(1.0, &pool);
    _ = Value.input(2.0, &pool);
    const b = Value.param(0.5, &pool);

    const p = pool.params();
    try std.testing.expectEqual(@as(usize, 2), p.len);
    try std.testing.expectEqual(@as(f32, 1.0), p[0].data);
    try std.testing.expectEqual(@as(f32, 0.5), p[1].data);
    _ = w;
    _ = b;
}
