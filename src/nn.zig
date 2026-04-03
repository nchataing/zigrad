const std = @import("std");
const engine = @import("engine.zig");
const Pool = engine.Pool;
const Value = engine.Value;

pub fn Neuron(comptime input_size: usize) type {
    return struct {
        const Self = @This();

        weights: [input_size]*Value,
        bias: *Value,

        pub fn init(pool: *Pool, rng: std.Random) Self {
            var weights: [input_size]*Value = undefined;
            const scale: f32 = (5.0 / 3.0) / @sqrt(@as(f32, @floatFromInt(input_size)));
            for (&weights) |*w| {
                w.* = Value.param(rng.floatNorm(f32) * scale, pool);
            }
            const bias = Value.param(0, pool);
            return Self{
                .weights = weights,
                .bias = bias,
            };
        }

        pub fn apply(self: *const Self, inputs: [input_size]*Value) *Value {
            var sum = self.bias;
            for (self.weights, inputs) |w, input| {
                sum = input.mul(w).add(sum);
            }
            return sum;
        }
    };
}

pub fn Layer(comptime input_size: usize, comptime output_size: usize) type {
    return struct {
        const Self = @This();

        neurons: [output_size]Neuron(input_size),

        pub fn init(pool: *Pool, rng: std.Random) Self {
            var neurons: [output_size]Neuron(input_size) = undefined;
            for (&neurons) |*n| {
                n.* = Neuron(input_size).init(pool, rng);
            }
            return Self{
                .neurons = neurons,
            };
        }

        pub fn apply(self: *const Self, inputs: [input_size]*Value) [output_size]*Value {
            var outputs: [output_size]*Value = undefined;
            for (&self.neurons, 0..) |n, i| {
                outputs[i] = n.apply(inputs);
            }
            return outputs;
        }
    };
}
