# Faza 1 — de la Streamlit la FastAPI + Blazor Server

Acest document explică, fișier cu fișier, ce s-a construit în Faza 1 a migrării și de ce, la nivel de logică exactă (nu doar "ce face", ci "cum face").

## 1. Contextul și motivul migrării

Aplicația originală (`legacy/streamlit_app/app.py` + `hmp.py`) era un singur script Streamlit care:
- încărca un model YOLOv8 prin SAHI (`AutoDetectionModel.from_pretrained(..., device=0)`) — **hardcodat pe GPU**, cădea pe orice mașină fără CUDA;
- avea logica de extragere a punctelor centrale ale vehiculelor (`coords()`) și logica de heatmap tactic (`calculate_tactical_heatmap`) **duplicate cuvânt cu cuvânt** între `app.py` și `hmp.py`, cu `hmp.py` folosind `device="cpu"` — o inconsistență reală, nu doar stilistică;
- nu avea `requirements.txt`, `.gitignore`, sau limite explicite de upload (README pretindea 2000 MB, dar Streamlit limitează implicit la 200 MB).

Direcția pe termen lung (discutată și aprobată de utilizator) e ca interfața și logica de orchestrare să treacă pe .NET, iar partea de ML (YOLOv8 + SAHI) să rămână în Python — pentru că bibliotecile de detecție/slicing nu au echivalent matur în .NET. Faza 1 face exact atât: separă aplicația în două procese care comunică prin HTTP.

```
┌─────────────────────────┐        HTTP (multipart/JSON)        ┌──────────────────────────┐
│   MilitaryTrack.Web      │ ───────────────────────────────────▶ │   inference_service        │
│   (Blazor Server, .NET)  │ ◀─────────────────────────────────── │   (FastAPI, Python 3.11)   │
│   UI + API gateway       │                                      │   YOLOv8 + SAHI            │
└─────────────────────────┘                                      └──────────────────────────┘
        │ referă
        ▼
┌─────────────────────────┐
│  MilitaryTrack.Core      │  DTO-uri + InferenceClient (HttpClient tipizat)
│  (class library, fără    │  — fără nicio dependență de Blazor, ca un viitor
│   dependențe Blazor)     │    CLI să-l poată referi direct.
└─────────────────────────┘
```

---

## 2. `inference_service/` — microserviciul Python (FastAPI)

Rulează separat, pe portul 8000 implicit. Singurul loc unde se încarcă modelul YOLO și unde rulează SAHI.

### 2.1 `app/config.py`

Definește `Settings` (pydantic-settings, citește din `.env`):

- `model_path`, `device` ("auto"/"cpu"/"cuda:0"), `model_load_confidence_floor` (0.05), praguri implicite (`default_confidence_threshold`, `default_slice_size`, `default_overlap_ratio`, `default_gsd_m_per_px`, `default_grid_size_m`).
- `resolve_model_path()` — dacă `MODEL_PATH` nu e setat explicit, caută în ordine: `inference_service/best.pt`, `<root>/best.pt`, `<root>/runs/detect/rezultat_militar/weights/best.pt` (exact fallback-ul din `app.py` original, dar centralizat într-un singur loc în loc să fie reimplementat).
- `resolve_device()` — dacă `device == "auto"`, face `torch.cuda.is_available()` și alege `"cuda:0"` sau `"cpu"`. **Asta rezolvă bug-ul `device=0` hardcodat.**
- `get_class_names()` — citește `config.yaml` (cel de la rădăcina proiectului, folosit și de `antrenare.py`) și returnează `{0: "Small_Military_Vehicle", ...}`, ca lista de clase să aibă o singură sursă de adevăr.

### 2.2 `app/detection.py`

Logica de inferență, extrasă din duplicarea `app.py`/`hmp.py` într-un singur loc:

- `load_model(settings)` — creează `AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=..., confidence_threshold=settings.model_load_confidence_floor, device=...)`.

  **Decizie de design importantă**: SAHI "coace" pragul de încredere (`confidence_threshold`) în model la momentul construcției — nu poate fi schimbat per-request fără să reîncarci modelul, ceea ce ar fi mult prea lent. Soluția: modelul se încarcă **o singură dată**, la pornire, cu un prag foarte jos (0.05, `model_load_confidence_floor`), ca să nu piardă din start predicții. Filtrarea la pragul cerut de utilizator se face **după** inferență, per-request, în `filter_by_confidence()`.

- `run_sliced_detection(image_np, model, slice_size, overlap_ratio)` — apelează `sahi.predict.get_sliced_prediction` cu `slice_height=slice_width=slice_size`, `overlap_height_ratio=overlap_width_ratio=overlap_ratio`. Identic cu ce făcea `app.py` (640×640, overlap 0.2 implicit).

- `filter_by_confidence(result, confidence_threshold)` — `[obj for obj in result.object_prediction_list if obj.score.value >= confidence_threshold]`. Acesta e post-filtrul care face pragul configurabil per-request.

- `get_class_counts(predictions)` — numără câte predicții sunt din fiecare clasă (`{class_name: count}`).

- `extract_vehicle_points(predictions, exclude_classes={"Civilian_Vehicle"})` — pentru fiecare predicție a cărei clasă **nu** e în `exclude_classes`, calculează centrul bbox-ului: `center_x = (x_min + x_max) / 2`, `center_y = (y_min + y_max) / 2`. Returnează un `np.ndarray` de forma `(N, 2)` (sau `(0, 2)` dacă nu sunt puncte). Aceasta e funcția care înlocuiește `coords()`/`coordinates()` duplicate din vechiul cod — vehiculele civile sunt excluse din harta tactică pentru că scopul e mișcarea de trupe/echipament militar, nu trafic civil.

### 2.3 `app/tactical.py`

Matematica hărții tactice, parametrizată (în loc de constante hardcodate la nivel de modul, ca înainte):

- `calculate_tactical_heatmap(points_t0, points_t1, img_w, img_h, gsd_m_per_px, grid_size_m)`:
  1. `pixelgrid = grid_size_m / gsd_m_per_px` — câți pixeli are o latură a unui sector de teren. Cu GSD implicit 0.3 m/pixel (rezoluția la sol a imaginii) și sectoare de 50m, `pixelgrid = 50 / 0.3 ≈ 166.67` pixeli.
  2. `bins_x = int(img_w / pixelgrid)`, `bins_y = int(img_h / pixelgrid)` — câte sectoare încap pe lățime/înălțime (minim 1, ca să nu împartă la zero mai departe).
  3. `np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[0, img_w], [0, img_h]])` rulat separat pentru punctele T0 și T1 — practic pune fiecare vehicul detectat în sectorul lui de 50×50m și numără câte sunt în fiecare sector.
  4. `diff_matrix = heatmap_t1 - heatmap_t0` — pentru fiecare sector, diferența de număr de vehicule între T1 și T0. Pozitiv = au apărut vehicule (sosire), negativ = au dispărut (plecare). **Exact aceeași logică ca în `app.py`/`hmp.py` originale**, doar că acum ia parametrii ca argumente în loc de constante globale.

- `calculate_density(vehicle_count, bins_x, bins_y, grid_size_m)` — suprafața totală analizată = `bins_x * bins_y * grid_size_m²`, convertită din m² în hectare (`/10000`); densitatea = `vehicle_count / total_area_ha`.

### 2.4 `app/schemas.py`

Modelele Pydantic pentru request/response (`HealthResponse`, `ConfigResponse`, `Detection`, `DetectResponse`, `TacticalMapResponse`). Acestea definesc exact forma JSON pe care o vede clientul .NET — vezi DTO-urile din `MilitaryTrack.Core` (secțiunea 3), care sunt oglinda lor în C#.

### 2.5 `app/routers/meta.py`

- `GET /api/v1/health` — citește `request.app.state.model`; dacă e `None` (modelul nu s-a putut încărca la pornire, de ex. nu există `best.pt`), returnează `status: "degraded"` cu detaliul erorii, altfel `status: "ok"` cu device-ul și calea modelului. Folosit de Blazor ca să afișeze un mesaj prietenos în loc de eroare neașteptată.
- `GET /api/v1/config` — expune lista de clase (din `config.yaml`) și pragurile implicite, ca frontend-ul să nu le re-hardcodeze.

### 2.6 `app/routers/detect.py` — `POST /api/v1/detect`

1. Primește `file` (imagine, multipart) + opțional `confidence_threshold`, `slice_size`, `overlap_ratio` (form fields; dacă lipsesc, se folosesc valorile implicite din `Settings`).
2. Dacă modelul nu e încărcat → `HTTPException(503, detail=...)`.
3. `Image.open(file.file).convert("RGB")` → `np.array(image)`.
4. `detection.run_sliced_detection(...)` → `detection.filter_by_confidence(...)`.
5. Construiește `DetectResponse`: dimensiuni imagine, `total_objects`, `counts` (toate clasele, inclusiv Civilian_Vehicle — spre deosebire de harta tactică, aici nu se exclude nimic), lista `detections` cu `class_name`, `confidence` (din `obj.score.value`), `bbox` (`obj.bbox.to_xyxy()` — `[x_min, y_min, x_max, y_max]` în pixeli originali).

### 2.7 `app/routers/tactical.py` — `POST /api/v1/tactical-map`

1. Primește `file_t0`, `file_t1` + opțional `confidence_threshold`, `gsd_m_per_px`, `grid_size_m`.
2. Deschide ambele imagini; dacă dimensiunile diferă, adaugă un mesaj în `warnings` (nu blochează procesarea — la fel ca avertismentul din Streamlit original).
3. Rulează detecția separat pe T0 și T1 (aceiași parametri de slicing impliciți pentru amândouă), filtrează după prag, extrage punctele vehiculelor (excluzând civile) pentru fiecare.
4. Cheamă `tactical.calculate_tactical_heatmap` și `tactical.calculate_density` (vezi 2.3).
5. Returnează `TacticalMapResponse` cu `diff_matrix` ca listă de liste (`diff_matrix[x][y]`), `bins_x`/`bins_y`, contoare de puncte, arie/densitate, și `warnings`.

### 2.8 `app/main.py`

- `lifespan()` — la pornirea serverului (nu la fiecare request), încearcă `load_model()`; dacă eșuează (nu găsește `best.pt`), salvează eroarea în `app.state.model_error` și lasă `app.state.model = None` în loc să crape serverul — serviciul rămâne "up" dar raportează `degraded` prin `/health`.
- `create_app()` — montează cele 3 routere sub prefixul `/api/v1`.

### 2.9 `examples/hmp_cli_example.py`

Înlocuiește `hmp.py`. Nu mai duplică logica — importă direct `app.detection` și `app.tactical` (fără HTTP), rulează pipeline-ul pe două fișiere date ca argumente de linie de comandă, printează rezultatele și opțional afișează heatmap-ul cu matplotlib/seaborn dacă sunt instalate (`requirements-dev.txt`). E marcat explicit ca script de sanity-check pentru dezvoltare, nu ca tool pentru utilizatori finali.

### 2.10 `requirements.txt` / `requirements-dev.txt`

`fastapi`, `uvicorn[standard]`, `python-multipart` (necesar pentru `File`/`Form`), `pydantic-settings`, `pyyaml`, `numpy`, `Pillow`, `ultralytics`, `sahi`, `torch`. **`matplotlib`/`seaborn` NU mai sunt în `requirements.txt`** — au fost mutate în `requirements-dev.txt` pentru că randarea heatmap-ului s-a mutat pe partea de Blazor (vezi 4.4), eliminând și problema de thread-safety a stării globale `pyplot` sub cereri concurente în FastAPI.

---

## 3. `src/MilitaryTrack.Core/` — biblioteca .NET partajată

Nu are nicio dependență de Blazor — doar `HttpClient` + DTO-uri + extensii de DI. Motivul: un viitor CLI (`MilitaryTrack.Cli`) va putea referi acest proiect direct și va folosi exact același `InferenceClient`.

### 3.1 `Dtos/*.cs`

Oglinda C# a schemelor Pydantic, cu `[JsonPropertyName("snake_case")]` explicit pe fiecare proprietate (FastAPI/Pydantic serializează în `snake_case`, C# convenția e `PascalCase`):

- `HealthStatus`, `ServiceConfig` — corespund `HealthResponse`/`ConfigResponse`.
- `DetectionOptions` (request, toate opționale — `double? ConfidenceThreshold`, etc.) / `Detection` / `DetectionResult` — corespund `DetectResponse`.
- `TacticalMapOptions` (request) / `TacticalMapResult` — corespund `TacticalMapResponse`. `DiffMatrix` e `double[][]`, indexat `[x][y]` — la fel ca JSON-ul din Python.

### 3.2 `InferenceServiceException.cs`

Excepție dedicată cu `StatusCode` — aruncată când serviciul Python răspunde cu eroare (ex. 503 dacă modelul nu e încărcat), ca paginile Blazor să poată distinge "serviciul a răspuns cu o eroare" de "nu am putut ajunge la serviciu deloc".

### 3.3 `InferenceClient.cs`

Client HTTP tipizat, cu 4 metode publice: `GetHealthAsync`, `GetConfigAsync`, `DetectAsync`, `GetTacticalMapAsync`.

**Detaliu important — streaming, nu buffering**: `AddFile()` construiește un `StreamContent` direct din `Stream`-ul primit ca parametru (nu îl citește într-un `byte[]` întâi). La fel, FastAPI primește fișierul ca `UploadFile`, susținut de un fișier temporar "spooled" pe partea Python — deci nu există un moment în care tot fișierul stă bufferat de două ori în memorie doar pentru transport (decodarea efectivă a imaginii pentru inferență tot are nevoie de array-ul complet în memorie o dată — asta e inerent la SAHI/PIL/numpy, nu ceva ce se poate evita).

`ReadOrThrowAsync<T>()` — dacă răspunsul nu e succes, încearcă să citească `{"detail": "..."}` din body (formatul standard FastAPI pentru erori) și aruncă `InferenceServiceException`; altfel deserializează JSON-ul normal cu `JsonSerializerOptions(JsonSerializerDefaults.Web)`.

### 3.4 `InferenceServiceOptions.cs` + `ServiceCollectionExtensions.cs`

`InferenceServiceOptions` — `BaseUrl` (implicit `http://localhost:8000`) și `TimeoutSeconds` (implicit 300 — inferența SAHI pe CPU pentru imagini mari poate dura mult, deci timeout-ul implicit de 100s al `HttpClient`-ului nu era suficient).

`AddMilitaryTrackInferenceClient(IServiceCollection, IConfiguration)` — leagă secțiunea de config `"InferenceService"` la `InferenceServiceOptions` și înregistrează `InferenceClient` ca typed HttpClient (`AddHttpClient<InferenceClient>(...)`), setând `BaseAddress` și `Timeout` din opțiuni. Orice host (Blazor azi, un CLI mâine) cheamă o singură linie ca să aibă clientul complet configurat.

---

## 4. `src/MilitaryTrack.Web/` — aplicația Blazor Server

### 4.1 `Program.cs`

- `AddRazorComponents().AddInteractiveServerComponents()` — activează modul Blazor Server (randare pe server, actualizări prin SignalR).
- `UploadOptions` (clasă definită tot în `Program.cs`, în namespace global) — citită din secțiunea `"Upload"` din `appsettings.json`, înregistrată ca singleton (injectată în `UploadPanel.razor`).
- `services.Configure<HubOptions>(o => o.MaximumReceiveMessageSize = uploadOptions.MaxFileSizeBytes)` — mărește limita implicită de mesaj SignalR (implicit ~32KB), pentru că upload-ul de fișiere în Blazor Server circulă prin circuitul SignalR, nu printr-un POST HTTP clasic.
- `AddMilitaryTrackInferenceClient(builder.Configuration)` — înregistrează clientul din `MilitaryTrack.Core`.

### 4.2 `appsettings.json`

```json
"InferenceService": { "BaseUrl": "http://localhost:8000", "TimeoutSeconds": 300 },
"Upload": { "MaxFileSizeBytes": 1073741824 }   // 1 GB
```
Înlocuiește pretenția nefundamentată de "2000 MB" din README-ul vechi cu o limită reală, configurată explicit și documentată.

### 4.3 `Components/Shared/UploadedImage.cs` + `UploadPanel.razor`

`UploadedImage` — un `record` simplu: `FileName`, `ContentType`, `Bytes` (`byte[]`). Are `DataUrl` (returnează `data:{ContentType};base64,{...}` pentru afișare directă în `<img src="...">`) și `OpenStream()` (un nou `MemoryStream` peste aceiași bytes, pentru a fi trimis la `InferenceClient`).

`UploadPanel.razor` — componentă reutilizată de ambele pagini:
1. `InputFile` cu `OnChange` → `HandleFileSelectedAsync`.
2. Verifică `file.Size` față de `UploadOptions.MaxFileSizeBytes` (injectat) — dacă depășește, afișează eroare fără să încerce citirea.
3. `file.OpenReadStream(maxAllowedSize)` (Blazor limitează implicit citirea la ~500KB dacă nu specifici `maxAllowedSize` explicit — de-aia trebuie pasat) → copiază în `MemoryStream` → construiește `UploadedImage`.
4. Notifică părintele prin `EventCallback<UploadedImage?> ImageChanged`.

**De ce se bufferează totuși bytes-ii aici**: ca să poți arăta preview-ul (`<img>`) *și* să poți retrimite aceiași bytes la API fără să ceri utilizatorului să reselecteze fișierul, trebuie păstrați undeva. Costul e comparabil cu ce făcea Streamlit-ul original (`Image.open(uploaded_file)` ținea deja tot fișierul în memorie pentru afișare).

### 4.4 `Components/Shared/DetectionOverlay.razor`

Randează cutiile de detecție ca un layer SVG peste `<img>`:
- `<svg viewBox="0 0 {ImageWidth} {ImageHeight}">` — pentru că `viewBox`-ul e în coordonatele *originale* ale imaginii (pixeli reali din răspunsul API), SVG-ul se scalează automat corect indiferent la ce dimensiune e afișată imaginea pe ecran (CSS `max-width: 100%`).
- Pentru fiecare detecție: `<rect>` cu `x, y, width, height` calculate din `bbox = [x_min, y_min, x_max, y_max]`, culoare aleasă din `Palette` (6 culori fixe), consistentă per clasă via `ColorByClass` (dicționar static — prima clasă întâlnită ia prima culoare, etc.).
- **Detaliu tehnic non-evident**: eticheta de text nu putea fi scrisă ca `<text>...</text>` direct în markup Razor — Razor tratează `<text>` ca un tag special (pseudo-element pentru text brut) și dă eroare de compilare (`RZ1023`) dacă are atribute. Soluția: `TextLabelMarkup()` construiește manual string-ul HTML al elementului `<text>` (cu `HtmlEncode` pe conținut, ca protecție minimă), randat prin `@((MarkupString)...)`.
- Grosimea liniei și mărimea fontului sunt proporționale cu dimensiunea imaginii (`Math.Max(width, height) / 500.0`, respectiv `/ 90.0`), ca detecțiile pe imagini foarte mari/mici să rămână lizibile.

### 4.5 `Components/Shared/DivergingHeatmap.razor`

Randează `diff_matrix` (de la `/api/v1/tactical-map`) ca grid SVG, echivalentul heatmap-ului `seaborn` din Streamlit original (`cmap="vlag", center=0, annot=True, fmt=".0f"`):
- Fiecare celulă `(x, y)` e un pătrat de `CellSize=40` px, poziționat la `(x*40, y*40)` în viewBox-ul SVG. **Nu e nevoie de transpunerea `.T` pe care o făcea seaborn** — aceea era necesară doar pentru convenția de randare rând/coloană a seaborn; aici se iterează direct pe `x` (coloane) și `y` (rânduri), care corespunde exact la `diff_matrix[x][y]`.
- `MaxAbs` = valoarea absolută maximă din toată matricea (minim 1.0, ca să nu împartă la zero). Fiecare valoare e normalizată la `[-1, 1]` prin `value / MaxAbs`.
- `ColorFor(value)` — interpolare liniară (`Lerp`) între alb `(255,255,255)` și roșu `(179,33,33)` pentru valori pozitive, respectiv alb și albastru `(33,90,179)` pentru valori negative — replicarea manuală a paletei diverging `vlag`.
- Text-ul din fiecare celulă (`value:F0`, rotunjit la întreg, ca `fmt=".0f"` din seaborn) e alb sau închis în funcție de cât de saturată e culoarea de fundal (`Math.Abs(value) > MaxAbs * 0.6`), ca să rămână lizibil. Aceeași soluție `MarkupString` ca la `DetectionOverlay` pentru `<text>`.
- Legenda de sub grid arată explicit ce înseamnă roșu/alb/albastru.

### 4.6 `Components/Pages/ObjectDetection.razor` (`/detect`)

1. La inițializare (`OnInitializedAsync`), cheamă `InferenceClient.GetConfigAsync()` ca să pre-completeze `ConfidenceThreshold` cu valoarea implicită a serviciului; dacă serviciul nu răspunde, prinde excepția și afișează un banner "Inference service unavailable" în loc să crape pagina.
2. `UploadPanel` → `OnImageChanged` reține imaginea selectată și resetează rezultatul/erorile anterioare.
3. Butonul "Detect Objects" (dezactivat dacă nu e imagine sau e deja în curs) → `DetectAsync()`: deschide stream din `UploadedImage`, cheamă `InferenceClient.DetectAsync(stream, fileName, options)`.
4. La succes: `DetectionOverlay` (cu imaginea + detecțiile) +, dacă `TotalObjects > 0`, un tabel cu numărătoarea pe clase (`_result.Counts`); altfel mesaj "No objects detected" (echivalentul exact al comportamentului din Streamlit original).
5. Erori: `InferenceServiceException` → mesaj specific de eroare de la API; orice altă excepție (ex. serviciul e complet jos) → mesaj generic "Could not reach the inference service".

### 4.7 `Components/Pages/TacticalMap.razor` (`/tactical-map`)

Aceeași structură, dar cu două `UploadPanel` (T0/T1) și `CanGenerate = _imageT0 is not null && _imageT1 is not null` care controlează dacă butonul e activ. La succes:
- Afișează fiecare mesaj din `_result.Warnings` (ex. dimensiuni diferite T0/T1) ca banner de avertisment — server-side, nu mai trebuie comparate dimensiunile pe client, API-ul deja face verificarea.
- 3 "metric cards" (Vehicule Detectate = `PointsT1Count`, Arie Analizată = `TotalAreaHa`, Densitate Medie = `DensityT1VehPerHa`) — echivalentul `st.metric` din Streamlit.
- Textul "Tactical Grid Resolution: NxM sectors" + `DivergingHeatmap`.

### 4.8 `Components/Layout/MainLayout.razor` + `NavMenu.razor`

Înlocuiesc bara laterală (`st.sidebar.radio`) din Streamlit: un `<nav class="sidebar">` fix, cu linkuri către `/detect` și `/tactical-map` (`NavLink`, care își aplică singur clasa `active` pe ruta curentă).

### 4.9 `Components/Pages/Home.razor` (`/`)

Pagină de start simplă, cu link-uri către cele două moduri — Streamlit nu avea o pagină de start separată (totul era pe un singur ecran cu switch în sidebar), acum fiecare mod are propriul URL, deci are sens o pagină de intrare minimă.

---

## 5. Fluxul complet al unei cereri (mod Object Detection)

1. Utilizatorul deschide `/detect` în browser → Blazor Server deschide un circuit SignalR.
2. `OnInitializedAsync` → `GET http://localhost:8000/api/v1/config` (către Python) → pragul de încredere din UI e pre-completat.
3. Utilizatorul alege un fișier → `InputFile` transmite bytes-ii prin circuitul SignalR (server-side, deci limita `MaximumReceiveMessageSize` contează) → `UploadPanel` îi citește într-un `MemoryStream` → `UploadedImage` cu preview afișat instant (base64 data URL, fără roundtrip la server pentru afișare).
4. Click pe "Detect Objects" → `InferenceClient.DetectAsync` construiește `multipart/form-data` cu `StreamContent` peste bytes-ii deja în memorie → `POST http://localhost:8000/api/v1/detect`.
5. Pe Python: `Image.open` → `np.array` → `get_sliced_prediction` (SAHI feliază imaginea în tile-uri de 640×640 cu overlap 20%, rulează YOLO pe fiecare tile, combină rezultatele) → filtrare după prag → serializare `DetectResponse` (JSON).
6. .NET deserializează JSON-ul în `DetectionResult` → `DetectionOverlay` desenează SVG-ul peste imaginea deja afișată (fără alt round-trip, fără PNG suplimentar de la server) → tabelul de numărători se populează din `_result.Counts`.

Fluxul pentru `/tactical-map` e identic, dublat pentru T0/T1, cu pasul suplimentar de calcul al matricei de diferență și densitate pe partea Python, și randarea `DivergingHeatmap` în loc de `DetectionOverlay`.

---

## 6. Ce NU s-a făcut încă (intenționat, în afara scopului Fazei 1)

- **`MilitaryTrack.Cli`** — un proiect console .NET care ar folosi exact același `InferenceClient` din `MilitaryTrack.Core`. Nu există încă niciun fișier pentru el; arhitectura actuală (Core fără dependențe Blazor) e pregătită să-l primească fără refactorizare.
- **Integrare Sentinel Hub / monitorizare "în timp real"** — Sentinel-2 (gratuit) are rezoluție de 10m/pixel, insuficientă pentru vehicule individuale (2-5m). Ar necesita un model diferit, antrenat pentru detecție de clustere mari/schimbări la scară mare, plus stocare geospațială (PostGIS) și job-uri de polling — nimic din toate astea nu există încă.
- **Docker / docker-compose** — nu sunt instalate pe mașina de dezvoltare curentă; pornirea locală se face cu două terminale (`uvicorn` + `dotnet run`), documentat în `README.md`.

## 7. Verificare făcută

Codul Python a fost testat funcțional (nu doar compilat) cu stub-uri pentru `sahi`/`torch` (fără cele două biblioteci grele instalate local): `/api/v1/health`, `/api/v1/config`, `/api/v1/detect` și `/api/v1/tactical-map` au fost apelate direct prin `TestClient`-ul FastAPI, verificând că filtrarea după prag, numărătoarea pe clase și calculul de heatmap/densitate produc exact rezultatele așteptate pentru un set de predicții fictive controlate.

Soluția .NET a fost compilată integral (`dotnet build MilitaryTrack.slnx`, 0 erori) și rulată live: cu serviciul Python (stub) pornit pe portul 8000 și Blazor pe 5095, am încărcat o imagine sintetică prin browser real (Chrome, prin automatizare), apăsat "Detect Objects" și confirmat vizual overlay-ul SVG + tabelul de numărători; la fel pentru `/tactical-map`, cu ambele imagini T0/T1, confirmând metric cards + heatmap corect randate.
