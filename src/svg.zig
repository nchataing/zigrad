const std = @import("std");
const Io = std.Io;

pub const Series = struct {
    values: []const f32,
    color: []const u8,
    label: []const u8,
};

pub fn write(
    writer: *Io.Writer,
    xs: []const f32,
    series: []const Series,
) !void {
    const width: f32 = 800;
    const height: f32 = 400;
    const margin: f32 = 40;

    // Find bounds
    var x_min: f32 = xs[0];
    var x_max: f32 = xs[0];
    var y_min: f32 = std.math.inf(f32);
    var y_max: f32 = -std.math.inf(f32);
    for (xs) |x| {
        x_min = @min(x_min, x);
        x_max = @max(x_max, x);
    }
    for (series) |s| {
        for (s.values) |y| {
            y_min = @min(y_min, y);
            y_max = @max(y_max, y);
        }
    }
    const y_pad = (y_max - y_min) * 0.05;
    y_min -= y_pad;
    y_max += y_pad;

    const total_w = width + 2 * margin;
    const total_h = height + 2 * margin;

    try writer.print(
        \\<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d:.0} {d:.0}">
        \\<rect width="100%" height="100%" fill="white"/>
        \\
    , .{ total_w, total_h });

    // Axes
    try writer.print(
        \\<line x1="{d:.0}" y1="{d:.0}" x2="{d:.0}" y2="{d:.0}" stroke="black" stroke-width="1"/>
        \\<line x1="{d:.0}" y1="{d:.0}" x2="{d:.0}" y2="{d:.0}" stroke="black" stroke-width="1"/>
        \\
    , .{
        margin, margin,          margin, margin + height,
        margin, margin + height, margin + width, margin + height,
    });

    // Series
    for (series) |s| {
        try writer.print("<polyline fill=\"none\" stroke=\"{s}\" stroke-width=\"2\" points=\"", .{s.color});
        for (xs, s.values) |x, y| {
            const px = margin + (x - x_min) / (x_max - x_min) * width;
            const py = margin + height - (y - y_min) / (y_max - y_min) * height;
            try writer.print("{d:.1},{d:.1} ", .{ px, py });
        }
        try writer.print("\"/>\n", .{});
    }

    // Legend
    for (series, 0..) |s, i| {
        const lx = margin + 10;
        const ly = margin + 20 + @as(f32, @floatFromInt(i)) * 20;
        try writer.print(
            \\<line x1="{d:.0}" y1="{d:.0}" x2="{d:.0}" y2="{d:.0}" stroke="{s}" stroke-width="2"/>
            \\<text x="{d:.0}" y="{d:.0}" font-size="12" font-family="sans-serif">{s}</text>
            \\
        , .{ lx, ly, lx + 20, ly, s.color, lx + 25, ly + 4, s.label });
    }

    try writer.print("</svg>\n", .{});
    try writer.flush();
}
