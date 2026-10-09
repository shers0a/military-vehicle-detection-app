using System.Net;
using System.Text;

namespace MilitaryTrack.Core.Tests;

public class InferenceClientTests
{
    private sealed class StubHandler(HttpStatusCode status, string json) : HttpMessageHandler
    {
        // Read during SendAsync: the client disposes the request content once the call returns.
        public string? LastRequestBody { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            LastRequestBody = request.Content is null ? null : await request.Content.ReadAsStringAsync(ct);
            return new HttpResponseMessage(status)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json"),
            };
        }
    }

    private static (InferenceClient Client, StubHandler Handler) CreateClient(HttpStatusCode status, string json)
    {
        var handler = new StubHandler(status, json);
        var http = new HttpClient(handler) { BaseAddress = new Uri("http://inference.test/") };
        return (new InferenceClient(http), handler);
    }

    private static MemoryStream FakeImage() => new([1, 2, 3]);

    [Fact]
    public async Task DetectAsync_MapsSnakeCaseJsonToDtos()
    {
        var (client, _) = CreateClient(HttpStatusCode.OK, """
            {
              "image_width": 200, "image_height": 100, "total_objects": 1,
              "counts": { "Armored_Fighting_Vehicle": 1 },
              "detections": [ { "class_name": "Armored_Fighting_Vehicle", "confidence": 0.9, "bbox": [10, 10, 30, 30] } ],
              "model_device": "cpu", "confidence_threshold_applied": 0.35
            }
            """);

        var result = await client.DetectAsync(FakeImage(), "a.png");

        Assert.Equal(200, result.ImageWidth);
        Assert.Equal(1, result.Counts["Armored_Fighting_Vehicle"]);
        Assert.Equal([10.0, 10, 30, 30], result.Detections.Single().Bbox);
        Assert.Equal("cpu", result.ModelDevice);
    }

    [Fact]
    public async Task DetectAsync_SendsOptionsAsInvariantCultureFormFields()
    {
        var (client, handler) = CreateClient(HttpStatusCode.OK, """{ "counts": {}, "detections": [] }""");

        await client.DetectAsync(FakeImage(), "a.png", new() { ConfidenceThreshold = 0.5, SliceSize = 512 });

        var body = handler.LastRequestBody!;
        Assert.Contains("name=confidence_threshold", body);
        Assert.Contains("0.5", body); // never "0,5", whatever the server's locale is
        Assert.Contains("name=slice_size", body);
        Assert.DoesNotContain("name=overlap_ratio", body); // unset options are not sent
    }

    [Fact]
    public async Task ErrorWithStringDetail_ThrowsWithThatMessageAndStatus()
    {
        var (client, _) = CreateClient(HttpStatusCode.ServiceUnavailable, """{ "detail": "No model weights found." }""");

        var ex = await Assert.ThrowsAsync<InferenceServiceException>(() => client.DetectAsync(FakeImage(), "a.png"));

        Assert.Equal(HttpStatusCode.ServiceUnavailable, ex.StatusCode);
        Assert.Equal("No model weights found.", ex.Message);
    }

    [Fact]
    public async Task ValidationError_IsFormattedAsFieldAndMessage()
    {
        var (client, _) = CreateClient(HttpStatusCode.UnprocessableEntity, """
            { "detail": [
                { "type": "less_than_equal", "loc": ["body", "confidence_threshold"], "msg": "Input should be less than or equal to 1" },
                { "type": "greater_than_equal", "loc": ["body", "slice_size"], "msg": "Input should be greater than or equal to 64" }
            ] }
            """);

        var ex = await Assert.ThrowsAsync<InferenceServiceException>(() => client.DetectAsync(FakeImage(), "a.png"));

        Assert.Equal(
            "confidence_threshold: Input should be less than or equal to 1; slice_size: Input should be greater than or equal to 64",
            ex.Message);
    }

    [Fact]
    public async Task ErrorWithoutJsonBody_FallsBackToReasonPhrase()
    {
        var (client, _) = CreateClient(HttpStatusCode.BadGateway, "<html>proxy error</html>");

        var ex = await Assert.ThrowsAsync<InferenceServiceException>(() => client.GetHealthAsync());

        Assert.Equal(HttpStatusCode.BadGateway, ex.StatusCode);
        Assert.Equal("Bad Gateway", ex.Message);
    }
}
