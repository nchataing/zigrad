const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addModule("zigrad", .{
        .root_source_file = b.path("src/engine.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = lib },
            },
        }),
    });

    b.installArtifact(exe);

    // zig build run [-- args]
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    // zig build test
    const test_step = b.step("test", "Run unit tests");
    const lib_tests = b.addTest(.{
        .root_module = lib,
    });
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);
}
