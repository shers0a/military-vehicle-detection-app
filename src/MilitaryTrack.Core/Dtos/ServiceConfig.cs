using System.Text.Json.Serialization;

namespace MilitaryTrack.Core.Dtos;

public sealed class ServiceConfig
{
    [JsonPropertyName("class_names")]
    public List<string> ClassNames { get; set; } = [];

    [JsonPropertyName("default_confidence_threshold")]
    public double DefaultConfidenceThreshold { get; set; }

    [JsonPropertyName("default_slice_size")]
    public int DefaultSliceSize { get; set; }

    [JsonPropertyName("default_overlap_ratio")]
    public double DefaultOverlapRatio { get; set; }

    [JsonPropertyName("default_gsd_m_per_px")]
    public double DefaultGsdMPerPx { get; set; }

    [JsonPropertyName("default_grid_size_m")]
    public double DefaultGridSizeM { get; set; }
}
