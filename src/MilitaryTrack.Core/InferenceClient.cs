using System.Net.Http.Json;
using System.Text.Json;
using MilitaryTrack.Core.Dtos;

namespace MilitaryTrack.Core;

/// <summary>Typed client for the Python FastAPI inference_service. Streams uploads directly
/// from the source stream instead of buffering whole images in memory.</summary>
public sealed class InferenceClient(HttpClient httpClient)
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public async Task<HealthStatus> GetHealthAsync(CancellationToken ct = default)
    {
        var response = await httpClient.GetAsync("api/v1/health", ct);
        return await ReadOrThrowAsync<HealthStatus>(response, ct);
    }

    public async Task<ServiceConfig> GetConfigAsync(CancellationToken ct = default)
    {
        var response = await httpClient.GetAsync("api/v1/config", ct);
        return await ReadOrThrowAsync<ServiceConfig>(response, ct);
    }

    public async Task<DetectionResult> DetectAsync(
        Stream imageStream,
        string fileName,
        DetectionOptions? options = null,
        CancellationToken ct = default)
    {
        using var content = new MultipartFormDataContent();
        AddFile(content, imageStream, fileName, "file");
        AddFormField(content, "confidence_threshold", options?.ConfidenceThreshold);
        AddFormField(content, "slice_size", options?.SliceSize);
        AddFormField(content, "overlap_ratio", options?.OverlapRatio);

        var response = await httpClient.PostAsync("api/v1/detect", content, ct);
        return await ReadOrThrowAsync<DetectionResult>(response, ct);
    }

    public async Task<TacticalMapResult> GetTacticalMapAsync(
        Stream imageT0Stream,
        string fileNameT0,
        Stream imageT1Stream,
        string fileNameT1,
        TacticalMapOptions? options = null,
        CancellationToken ct = default)
    {
        using var content = new MultipartFormDataContent();
        AddFile(content, imageT0Stream, fileNameT0, "file_t0");
        AddFile(content, imageT1Stream, fileNameT1, "file_t1");
        AddFormField(content, "confidence_threshold", options?.ConfidenceThreshold);
        AddFormField(content, "gsd_m_per_px", options?.GsdMPerPx);
        AddFormField(content, "grid_size_m", options?.GridSizeM);

        var response = await httpClient.PostAsync("api/v1/tactical-map", content, ct);
        return await ReadOrThrowAsync<TacticalMapResult>(response, ct);
    }

    private static void AddFile(MultipartFormDataContent content, Stream stream, string fileName, string fieldName)
    {
        var streamContent = new StreamContent(stream);
        content.Add(streamContent, fieldName, fileName);
    }

    private static void AddFormField(MultipartFormDataContent content, string name, double? value)
    {
        if (value.HasValue)
        {
            content.Add(new StringContent(value.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)), name);
        }
    }

    private static void AddFormField(MultipartFormDataContent content, string name, int? value)
    {
        if (value.HasValue)
        {
            content.Add(new StringContent(value.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)), name);
        }
    }

    private static async Task<T> ReadOrThrowAsync<T>(HttpResponseMessage response, CancellationToken ct)
    {
        if (!response.IsSuccessStatusCode)
        {
            string detail;
            try
            {
                var problem = await response.Content.ReadFromJsonAsync<JsonElement>(ct);
                detail = problem.TryGetProperty("detail", out var d) ? FormatDetail(d) : response.ReasonPhrase ?? "Request failed";
            }
            catch
            {
                detail = response.ReasonPhrase ?? "Request failed";
            }

            throw new InferenceServiceException(detail, response.StatusCode);
        }

        var result = await response.Content.ReadFromJsonAsync<T>(JsonOptions, ct);
        return result ?? throw new InferenceServiceException("Empty response from inference service", response.StatusCode);
    }

    /// <summary>FastAPI returns <c>detail</c> as a plain string for errors raised in code, but as a
    /// list of <c>{loc, msg}</c> objects for request validation errors (422).</summary>
    private static string FormatDetail(JsonElement detail)
    {
        if (detail.ValueKind != JsonValueKind.Array)
        {
            return detail.ToString();
        }

        return string.Join("; ", detail.EnumerateArray().Select(error =>
        {
            var field = error.TryGetProperty("loc", out var loc) && loc.ValueKind == JsonValueKind.Array && loc.GetArrayLength() > 0
                ? loc[loc.GetArrayLength() - 1].ToString()
                : "request";
            var message = error.TryGetProperty("msg", out var msg) ? msg.ToString() : error.ToString();
            return $"{field}: {message}";
        }));
    }
}
