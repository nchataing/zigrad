const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addModule("zigrad", .{
        .root_source_file = b.path("src/engine.zig"),
        .target = target,
        .optimize = optimize,
    });

    // zig build test
    const test_step = b.step("test", "Run unit tests");
    const lib_tests = b.addTest(.{
        .root_module = lib,
    });
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);
}
