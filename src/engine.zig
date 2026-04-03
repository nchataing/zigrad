const std = @import("std");
pub const nn = @import("nn.zig");
pub const svg = @import("svg.zig");

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
            self.freeNode(node);
        }
        for (self.param_nodes.items) |node| {
            self.freeNode(node);
        }
        self.nodes.deinit(self.allocator);
        self.param_nodes.deinit(self.allocator);
    }

    pub fn create(self: *Pool, value: Value) *Value {
        const node = self.allocator.create(Value) catch @panic("OOM");
        node.* = value;
        node.pool = self;
        if (value.role == .param) {
            self.param_nodes.append(self.allocator, node) catch @panic("OOM");
        } else {
            self.nodes.append(self.allocator, node) catch @panic("OOM");
        }
        return node;
    }

    pub fn clearInternal(self: *Pool) void {
        for (self.nodes.items) |node| {
            self.freeNode(node);
        }
        self.nodes.clearRetainingCapacity();
    }

    pub fn params(self: *Pool) []*Value {
        return self.param_nodes.items;
    }

    fn freeNode(self: *Pool, node: *Value) void {
        if (node._children.len > 0) {
            self.allocator.free(node._children);
        }
        self.allocator.destroy(node);
    }

    /// Allocate a children slice on the pool's allocator.
    fn allocChildren(self: *Pool, children: anytype) []*Value {
        const slice = self.allocator.alloc(*Value, children.len) catch @panic("OOM");
        inline for (0..children.len) |i| {
            slice[i] = children[i];
        }
        return slice;
    }
};

pub const Value = struct {
    data: f32,
    grad: f32 = 0,
    role: Role = .internal,
    pool: *Pool = undefined,
    _children: []*Value = &.{},
    _backward: ?*const fn (self: *Value) void = null,

    pub const Role = enum { param, input, internal };

    pub fn param(data: f32, pool: *Pool) *Value {
        return pool.create(.{ .data = data, .role = .param });
    }

    pub fn input(data: f32, pool: *Pool) *Value {
        return pool.create(.{ .data = data, .role = .input });
    }

    pub fn add(self: *Value, other: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].grad += v.grad;
                v._children[1].grad += v.grad;
            }
        }.backward;
        return self.pool.create(.{
            .data = self.data + other.data,
            ._children = self.pool.allocChildren(.{ self, other }),
            ._backward = &_backward,
        });
    }

    pub fn sub(self: *Value, other: *Value) *Value {
        return self.add(other.neg());
    }

    pub fn neg(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].grad += -v.grad;
            }
        }.backward;
        return self.pool.create(.{
            .data = -self.data,
            ._children = self.pool.allocChildren(.{self}),
            ._backward = &_backward,
        });
    }

    pub fn mul(self: *Value, other: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].grad += v.grad * v._children[1].data;
                v._children[1].grad += v.grad * v._children[0].data;
            }
        }.backward;
        return self.pool.create(.{
            .data = self.data * other.data,
            ._children = self.pool.allocChildren(.{ self, other }),
            ._backward = &_backward,
        });
    }

    pub fn pow(self: *Value, exponent: f32) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                const e = v._children[1].data;
                v._children[0].grad += v.grad * e * std.math.pow(f32, v._children[0].data, e - 1);
            }
        }.backward;
        const exp_node = self.pool.create(.{ .data = exponent });
        return self.pool.create(.{
            .data = std.math.pow(f32, self.data, exponent),
            ._children = self.pool.allocChildren(.{ self, exp_node }),
            ._backward = &_backward,
        });
    }

    pub fn exp(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].grad += v.grad * v.data;
            }
        }.backward;
        return self.pool.create(.{
            .data = @exp(self.data),
            ._children = self.pool.allocChildren(.{self}),
            ._backward = &_backward,
        });
    }

    pub fn log(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                v._children[0].grad += v.grad / v._children[0].data;
            }
        }.backward;
        return self.pool.create(.{
            .data = @log(self.data),
            ._children = self.pool.allocChildren(.{self}),
            ._backward = &_backward,
        });
    }

    pub fn tanh(self: *Value) *Value {
        const _backward = struct {
            fn backward(v: *Value) void {
                const out = v.data;
                v._children[0].grad += v.grad * (1 - out * out);
            }
        }.backward;
        return self.pool.create(.{
            .data = std.math.tanh(self.data),
            ._children = self.pool.allocChildren(.{self}),
            ._backward = &_backward,
        });
    }

    /// Fused cross-entropy loss: -log(softmax(logits)[target])
    /// Gradient: softmax_i - 1_{i==target}
    pub fn crossEntropyLoss(logits: []*Value, target: usize) *Value {
        const pool = logits[0].pool;

        // Forward: numerically stable softmax + NLL
        var max: f32 = -std.math.inf(f32);
        for (logits) |l| max = @max(max, l.data);

        var sum_exp: f32 = 0;
        for (logits) |l| sum_exp += @exp(l.data - max);

        const loss = -logits[target].data + max + @log(sum_exp);

        // Children: all logits + a phantom node storing the target index
        const children = pool.allocator.alloc(*Value, logits.len + 1) catch @panic("OOM");
        @memcpy(children[0..logits.len], logits);
        children[logits.len] = pool.create(.{ .data = @floatFromInt(target) });

        const _backward = struct {
            fn backward(v: *Value) void {
                const n = v._children.len - 1;
                const tgt: usize = @intFromFloat(v._children[n].data);

                var m: f32 = -std.math.inf(f32);
                for (v._children[0..n]) |c| m = @max(m, c.data);

                var s: f32 = 0;
                for (v._children[0..n]) |c| s += @exp(c.data - m);

                for (v._children[0..n], 0..) |c, i| {
                    const p = @exp(c.data - m) / s;
                    c.grad += v.grad * (p - if (i == tgt) @as(f32, 1.0) else @as(f32, 0.0));
                }
            }
        }.backward;

        return pool.create(.{
            .data = loss,
            ._children = children,
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
        for (v._children) |child| {
            buildTopo(child, visited, topo, allocator);
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
    try std.testing.expectEqual(@as(f32, 6.0), a.grad);
}

test "tanh backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    const a = Value.param(0.0, &pool);
    const b = a.tanh();

    b.backward();

    try std.testing.expectEqual(@as(f32, 0.0), b.data);
    try std.testing.expectEqual(@as(f32, 1.0), a.grad);
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

test "cross-entropy loss backward" {
    var pool = Pool.init(std.testing.allocator);
    defer pool.deinit();

    // 3 logits, target = 1
    const a = Value.param(1.0, &pool);
    const b = Value.param(2.0, &pool);
    const c = Value.param(0.5, &pool);

    var logits = [_]*Value{ a, b, c };
    const loss = Value.crossEntropyLoss(&logits, 1);

    loss.backward();

    // softmax([1, 2, 0.5]) = [exp(1), exp(2), exp(0.5)] / sum
    const ea = @exp(@as(f32, 1.0));
    const eb = @exp(@as(f32, 2.0));
    const ec = @exp(@as(f32, 0.5));
    const s = ea + eb + ec;

    // loss = -log(softmax[1]) = -log(exp(2)/s) = -2 + log(s)
    try std.testing.expectApproxEqAbs(-2.0 + @log(s), loss.data, 1e-5);

    // grads: softmax_i - 1_{i==target}
    try std.testing.expectApproxEqAbs(ea / s, a.grad, 1e-5);
    try std.testing.expectApproxEqAbs(eb / s - 1.0, b.grad, 1e-5);
    try std.testing.expectApproxEqAbs(ec / s, c.grad, 1e-5);
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
