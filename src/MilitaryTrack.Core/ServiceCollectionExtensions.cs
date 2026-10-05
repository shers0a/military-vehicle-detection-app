using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace MilitaryTrack.Core;

public static class ServiceCollectionExtensions
{
    /// <summary>Registers a typed HttpClient-backed InferenceClient, configured from the
    /// "InferenceService" configuration section (BaseUrl, TimeoutSeconds). Has no Blazor
    /// dependencies, so any host (Blazor Server today, a future CLI) can call this the same way.</summary>
    public static IServiceCollection AddMilitaryTrackInferenceClient(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<InferenceServiceOptions>(configuration.GetSection(InferenceServiceOptions.SectionName));

        services.AddHttpClient<InferenceClient>((provider, client) =>
        {
            var options = provider.GetRequiredService<Microsoft.Extensions.Options.IOptions<InferenceServiceOptions>>().Value;
            client.BaseAddress = new Uri(options.BaseUrl.TrimEnd('/') + "/");
            client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
        });

        return services;
    }
}
