# Build from the repository root (it also needs src/MilitaryTrack.Core):
#   podman build -f src/MilitaryTrack.Web/Dockerfile -t militarytrack-web .

# Stage 1: the full SDK compiles and publishes the app.
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Restore first, from the project files only, so the NuGet layer stays cached
# until a .csproj changes rather than on every code edit.
COPY src/MilitaryTrack.Core/MilitaryTrack.Core.csproj src/MilitaryTrack.Core/
COPY src/MilitaryTrack.Web/MilitaryTrack.Web.csproj src/MilitaryTrack.Web/
RUN dotnet restore src/MilitaryTrack.Web/MilitaryTrack.Web.csproj

COPY src/ src/
RUN dotnet publish src/MilitaryTrack.Web/MilitaryTrack.Web.csproj -c Release -o /app --no-restore

# Stage 2: only the ASP.NET runtime and the published output end up in the final image.
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .

ENV ASPNETCORE_HTTP_PORTS=8080
EXPOSE 8080
USER $APP_UID
ENTRYPOINT ["dotnet", "MilitaryTrack.Web.dll"]
