using System.Text.Json.Serialization;

namespace MilitaryTrack.Core.Dtos;

public sealed class HealthStatus
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("model_loaded")]
    public bool ModelLoaded { get; set; }

    [JsonPropertyName("device")]
    public string? Device { get; set; }

    [JsonPropertyName("model_path")]
    public string? ModelPath { get; set; }

    [JsonPropertyName("detail")]
    public string? Detail { get; set; }
}
