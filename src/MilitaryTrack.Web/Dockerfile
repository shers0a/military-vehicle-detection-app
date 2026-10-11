# Build from the repository root (it also needs src/MilitaryTrack.Core):
#   podman build -f src/MilitaryTrack.Web/Dockerfile -t militarytrack-web .

# Stage 1: the full SDK compiles and publishes the app.
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy the sources *before* restoring. The usual trick of restoring from the .csproj files
# alone (for layer caching) breaks Blazor here: the SDK only adds the package that ships
# _framework/blazor.web.js when it sees Razor components at restore time, so the page
# would load but never become interactive.
COPY src/ src/
RUN dotnet publish src/MilitaryTrack.Web/MilitaryTrack.Web.csproj -c Release -o /app

# Stage 2: only the ASP.NET runtime and the published output end up in the final image.
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .

ENV ASPNETCORE_HTTP_PORTS=8080
EXPOSE 8080
USER $APP_UID
ENTRYPOINT ["dotnet", "MilitaryTrack.Web.dll"]
