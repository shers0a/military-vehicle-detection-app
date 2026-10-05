namespace MilitaryTrack.Core;

public sealed class InferenceServiceOptions
{
    public const string SectionName = "InferenceService";

    public string BaseUrl { get; set; } = "http://localhost:8000";

    /// <summary>CPU-side SAHI sliced inference on large images can take a while;
    /// keep this generous and configurable rather than relying on HttpClient's 100s default.</summary>
    public int TimeoutSeconds { get; set; } = 300;
}
