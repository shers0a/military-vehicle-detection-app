namespace MilitaryTrack.Core;

/// <summary>Thrown when the inference_service returns a non-success response (e.g. model not loaded, 503).</summary>
public sealed class InferenceServiceException(string message, System.Net.HttpStatusCode statusCode) : Exception(message)
{
    public System.Net.HttpStatusCode StatusCode { get; } = statusCode;
}
