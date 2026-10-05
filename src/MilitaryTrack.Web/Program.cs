using MilitaryTrack.Core;
using MilitaryTrack.Web.Components;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Blazor Server file uploads are transported over the SignalR circuit; raise the
// default message size limit so large satellite imagery isn't rejected mid-upload.
// Actual enforcement of the per-upload ceiling happens via UploadOptions/InputFile's
// maxAllowedSize in UploadPanel.razor.
var uploadOptions = builder.Configuration.GetSection("Upload").Get<UploadOptions>() ?? new UploadOptions();
builder.Services.AddSingleton(uploadOptions);
builder.Services.Configure<Microsoft.AspNetCore.SignalR.HubOptions>(options =>
{
    options.MaximumReceiveMessageSize = uploadOptions.MaxFileSizeBytes;
});

builder.Services.AddMilitaryTrackInferenceClient(builder.Configuration);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseHttpsRedirection();

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();

public sealed class UploadOptions
{
    public long MaxFileSizeBytes { get; set; } = 1024L * 1024 * 1024;
}
