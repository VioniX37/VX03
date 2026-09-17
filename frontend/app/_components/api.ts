import type { Benchmarks, MLRecommendation, ModelMetadata, Preset, PresolveData, SolveResult, SystemInfo } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(`cannot reach the solver api at ${API_BASE}. start it with: python run_demo.py`);
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  systemInfo: () => request<SystemInfo>("/api/system_info"),
  presets: () => request<Preset[]>("/api/presets"),
  loadPreset: (presetId: string) => request<ModelMetadata>("/api/load_preset", postJson({ preset_id: presetId })),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ModelMetadata>("/api/upload_model", { method: "POST", body: form });
  },
  benchmarks: () => request<Benchmarks>("/api/benchmarks"),
  presolve: () => request<PresolveData>("/api/presolve", { method: "POST" }),
  recommend: () => request<MLRecommendation>("/api/ml_recommend", { method: "POST" }),
  solve: (algorithm: string, enablePresolve: boolean, timeLimitSeconds = 60) =>
    request<SolveResult>(
      "/api/solve",
      postJson({ algorithm, enable_presolve: enablePresolve, time_limit_seconds: timeLimitSeconds }),
    ),
};
