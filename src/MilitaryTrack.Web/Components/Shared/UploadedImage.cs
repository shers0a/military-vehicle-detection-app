namespace MilitaryTrack.Web.Components.Shared;

public sealed record UploadedImage(string FileName, string ContentType, byte[] Bytes)
{
    public string DataUrl => $"data:{ContentType};base64,{Convert.ToBase64String(Bytes)}";

    public Stream OpenStream() => new MemoryStream(Bytes, writable: false);
}
