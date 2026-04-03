const std = @import("std");
const Io = std.Io;
const zigrad = @import("zigrad");
const Pool = zigrad.Pool;
const Value = zigrad.Value;
const Layer = zigrad.nn.Layer;
const svg = zigrad.svg;

const n_samples = 50;
const epochs = 2000;

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buf);
    const stdout = &stdout_writer.interface;

    const allocator = std.heap.page_allocator;
    var pool = Pool.init(allocator);
    defer pool.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // MLP: 1 -> 8 -> 8 -> 1
    var layer1 = Layer(1, 8).init(&pool, rng);
    var layer2 = Layer(8, 8).init(&pool, rng);
    var layer3 = Layer(8, 1).init(&pool, rng);

    // Training data: sin(x) for x in [-pi, pi]
    var xs: [n_samples]f32 = undefined;
    var ys: [n_samples]f32 = undefined;
    for (0..n_samples) |i| {
        const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_samples - 1));
        xs[i] = -std.math.pi + t * 2.0 * std.math.pi;
        ys[i] = @sin(xs[i]);
    }

    for (0..epochs) |epoch| {
        const et: f32 = @as(f32, @floatFromInt(epoch)) / @as(f32, @floatFromInt(epochs - 1));
        const lr: f32 = 0.1 * std.math.pow(f32, 0.1, et); // 0.1 -> 0.01

        pool.clearInternal();

        var loss_node: ?*Value = null;

        for (0..n_samples) |i| {
            const input = [1]*Value{Value.input(xs[i], &pool)};

            // Forward: layer1 -> tanh -> layer2 -> tanh -> layer3
            var h1 = layer1.apply(input);
            for (&h1) |*v| v.* = v.*.tanh();
            var h2 = layer2.apply(h1);
            for (&h2) |*v| v.* = v.*.tanh();
            const out = layer3.apply(h2);

            // MSE: (pred - target)^2
            const target = Value.input(ys[i], &pool);
            const diff = out[0].sub(target);
            const sq = diff.mul(diff);

            loss_node = if (loss_node) |l| l.add(sq) else sq;
        }

        const loss = loss_node.?;
        loss.backward();

        // SGD update
        const n: f32 = @floatFromInt(n_samples);
        for (pool.params()) |p| {
            p.data -= lr * p.grad / n;
        }

        if (epoch % 10 == 0) {
            std.debug.print("epoch {d}: loss = {d:.6}\n", .{ epoch, loss.data / n });
        }
    }

    // Generate predictions for SVG
    pool.clearInternal();
    var predicted: [n_samples]f32 = undefined;
    for (0..n_samples) |i| {
        const input = [1]*Value{Value.input(xs[i], &pool)};
        var h1 = layer1.apply(input);
        for (&h1) |*v| v.* = v.*.tanh();
        var h2 = layer2.apply(h1);
        for (&h2) |*v| v.* = v.*.tanh();
        const out = layer3.apply(h2);
        predicted[i] = out[0].data;
    }

    try svg.write(stdout, &xs, &.{
        .{ .values = &ys, .color = "blue", .label = "sin(x)" },
        .{ .values = &predicted, .color = "red", .label = "predicted" },
    });
}
