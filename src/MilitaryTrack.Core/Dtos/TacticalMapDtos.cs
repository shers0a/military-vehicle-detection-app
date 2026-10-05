using System.Text.Json.Serialization;

namespace MilitaryTrack.Core.Dtos;

public sealed class TacticalMapOptions
{
    public double? ConfidenceThreshold { get; set; }
    public double? GsdMPerPx { get; set; }
    public double? GridSizeM { get; set; }
}

public sealed class TacticalMapResult
{
    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }

    [JsonPropertyName("bins_x")]
    public int BinsX { get; set; }

    [JsonPropertyName("bins_y")]
    public int BinsY { get; set; }

    /// <summary>diff_matrix[x][y] = count(T1) - count(T0) for that grid cell.</summary>
    [JsonPropertyName("diff_matrix")]
    public double[][] DiffMatrix { get; set; } = [];

    [JsonPropertyName("points_t0_count")]
    public int PointsT0Count { get; set; }

    [JsonPropertyName("points_t1_count")]
    public int PointsT1Count { get; set; }

    [JsonPropertyName("total_area_ha")]
    public double TotalAreaHa { get; set; }

    [JsonPropertyName("density_t1_veh_per_ha")]
    public double DensityT1VehPerHa { get; set; }

    [JsonPropertyName("gsd_m_per_px")]
    public double GsdMPerPx { get; set; }

    [JsonPropertyName("grid_size_m")]
    public double GridSizeM { get; set; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; set; } = [];
}
