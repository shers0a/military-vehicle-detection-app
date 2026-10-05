using System.Text.Json.Serialization;

namespace MilitaryTrack.Core.Dtos;

public sealed class DetectionOptions
{
    public double? ConfidenceThreshold { get; set; }
    public int? SliceSize { get; set; }
    public double? OverlapRatio { get; set; }
}

public sealed class Detection
{
    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = "";

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    /// <summary>[x_min, y_min, x_max, y_max] in original image pixel coordinates.</summary>
    [JsonPropertyName("bbox")]
    public double[] Bbox { get; set; } = [];
}

public sealed class DetectionResult
{
    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }

    [JsonPropertyName("total_objects")]
    public int TotalObjects { get; set; }

    [JsonPropertyName("counts")]
    public Dictionary<string, int> Counts { get; set; } = [];

    [JsonPropertyName("detections")]
    public List<Detection> Detections { get; set; } = [];

    [JsonPropertyName("model_device")]
    public string ModelDevice { get; set; } = "";

    [JsonPropertyName("confidence_threshold_applied")]
    public double ConfidenceThresholdApplied { get; set; }
}
