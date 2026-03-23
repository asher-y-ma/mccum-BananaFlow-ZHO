import { GoogleGenAI, Modality } from '@google/genai';
import type { GenerateContentResponse } from '@google/genai';

// 构建时由 Vite 注入：见 .env / .env.local 中 GEMINI_API_KEY、GEMINI_API_BASE_URL
const API_KEY = process.env.API_KEY;
const API_BASE_URL = process.env.GEMINI_API_BASE_URL;
if (!API_KEY) {
  console.warn("API_KEY environment variable not set. Using a placeholder. App will not function correctly.");
}
const ai = new GoogleGenAI({
  apiKey: API_KEY || "YOUR_API_KEY_HERE",
  ...(API_BASE_URL
    ? { httpOptions: { baseUrl: API_BASE_URL } }
    : {}),
});


const MAX_RETRIES = 3;
const INITIAL_DELAY_MS = 2000;

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * A higher-order function that wraps an API call with retry logic for rate limiting errors.
 */
const withRetry = <T extends (...args: any[]) => Promise<any>>(apiCall: T): T => {
    return (async (...args: Parameters<T>): Promise<ReturnType<T>> => {
        let lastError: any;
        for (let i = 0; i < MAX_RETRIES; i++) {
            try {
                return await apiCall(...args);
            } catch (error: any) {
                lastError = error;
                let isRateLimitError = false;

                if (error instanceof Error && error.message) {
                    try {
                        const errorDetails = JSON.parse(error.message);
                        if (errorDetails?.error?.code === 429 || errorDetails?.error?.status === 'RESOURCE_EXHAUSTED') {
                            isRateLimitError = true;
                        }
                    } catch (e) {
                        if (error.message.includes('429') || error.message.toLowerCase().includes('rate limit')) {
                           isRateLimitError = true;
                        }
                    }
                }
                
                if (isRateLimitError) {
                    if (i < MAX_RETRIES - 1) {
                        const backoffTime = INITIAL_DELAY_MS * Math.pow(2, i);
                        const jitter = Math.random() * 1000;
                        const waitTime = backoffTime + jitter;
                        console.warn(`Rate limit exceeded. Retrying in ${Math.round(waitTime / 1000)}s... (Attempt ${i + 1}/${MAX_RETRIES})`);
                        await delay(waitTime);
                        continue;
                    } else {
                        console.error(`API call failed after ${MAX_RETRIES} attempts due to rate limiting.`);
                    }
                }
                
                throw lastError;
            }
        }
        throw lastError;
    }) as T;
};

/**
 * Parses potential JSON error messages from the Gemini API for better user feedback.
 */
const formatError = (error: any, context: string): string => {
    console.error(`Error in ${context}:`, error);
    let errorMessage = error instanceof Error ? error.message : String(error);
    try {
        const parsedError = JSON.parse(errorMessage);
        if (parsedError?.error?.message) {
            errorMessage = parsedError.error.message;
            if (parsedError.error.status) {
                errorMessage += ` (Status: ${parsedError.error.status})`;
            }
        }
    } catch (e) {
        // Not a JSON string, use the message as is.
    }
    return `Error: ${errorMessage}`;
};

const generateContentWithRetry = withRetry(ai.models.generateContent);
const generateVideosWithRetry = withRetry(ai.models.generateVideos);
const getVideosOperationWithRetry = withRetry(ai.operations.getVideosOperation);

// Helper to convert File object to base64
const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};

// Helper to convert base64 string to a generative part
const base64ToGenerativePart = (base64: string, mimeType: string) => {
    return {
        inlineData: { data: base64, mimeType },
    };
};

/** Raw API / SDK may use camelCase or snake_case */
type InlineDataLike = {
    data?: string;
    mimeType?: string;
    mime_type?: string;
};

type FileDataLike = {
    fileUri?: string;
    file_uri?: string;
    mimeType?: string;
    mime_type?: string;
};

type PartLike = {
    text?: string;
    inlineData?: InlineDataLike;
    inline_data?: InlineDataLike;
    fileData?: FileDataLike;
    file_data?: FileDataLike;
};

const getInlineData = (part: PartLike): InlineDataLike | undefined =>
    part.inlineData ?? part.inline_data;

const getFileData = (part: PartLike): FileDataLike | undefined =>
    part.fileData ?? part.file_data;

const mimeFromInline = (d: InlineDataLike | undefined, fallback: string) =>
    d?.mimeType ?? d?.mime_type ?? fallback;

const mimeFromFile = (d: FileDataLike | undefined, fallback: string) =>
    d?.mimeType ?? d?.mime_type ?? fallback;

const isProbablyHttpUrl = (s: string) => /^https?:\/\//i.test(s.trim());

const parseDataUrl = (dataUrl: string): { base64: string; mimeType: string } | null => {
    const m = dataUrl.trim().match(/^data:([^;,]+);base64,(.+)$/s);
    if (!m) return null;
    return { mimeType: m[1] || 'image/png', base64: m[2] };
};

const blobToBase64 = (blob: Blob): Promise<string> =>
    new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const r = reader.result as string;
            const idx = r.indexOf(',');
            resolve(idx >= 0 ? r.slice(idx + 1) : r);
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
    });

/**
 * Fetch remote image (GCS signed URL, etc.) and return raw base64 + mime type.
 */
const fetchRemoteImageAsBase64 = async (url: string): Promise<{ base64: string; mimeType: string }> => {
    const res = await fetch(url, { mode: 'cors', credentials: 'omit' });
    if (!res.ok) {
        throw new Error(`Failed to fetch image (${res.status}): ${url.slice(0, 80)}…`);
    }
    const blob = await res.blob();
    const mimeType = blob.type && blob.type !== 'application/octet-stream'
        ? blob.type
        : 'image/png';
    const base64 = await blobToBase64(blob);
    return { base64, mimeType };
};

const tryTextAsImage = async (text: string): Promise<{ base64: string; mimeType: string } | null> => {
    const t = text.trim();
    if (!t) return null;
    const asData = parseDataUrl(t);
    if (asData) return asData;
    if (isProbablyHttpUrl(t)) {
        return fetchRemoteImageAsBase64(t);
    }
    return null;
};

/**
 * Gemini image models may return:
 * - inlineData: base64 image bytes
 * - fileData.fileUri: GCS or other HTTPS URL (must fetch)
 * - text: duplicate URL, or data URL, or plain text (ignored for image)
 */
export const extractImageFromGenerateContentParts = async (
    parts: PartLike[] | undefined | null,
): Promise<{ base64: string; mimeType: string } | null> => {
    if (!parts?.length) return null;

    for (const raw of parts) {
        const part = raw as PartLike;
        const inline = getInlineData(part);
        if (inline?.data) {
            return {
                base64: inline.data,
                mimeType: mimeFromInline(inline, 'image/png'),
            };
        }
    }

    for (const raw of parts) {
        const part = raw as PartLike;
        const fd = getFileData(part);
        const uri = fd?.fileUri ?? fd?.file_uri;
        if (uri && isProbablyHttpUrl(uri)) {
            try {
                const fetched = await fetchRemoteImageAsBase64(uri);
                return {
                    base64: fetched.base64,
                    mimeType: mimeFromFile(fd, fetched.mimeType),
                };
            } catch {
                /* try next part */
            }
        }
    }

    for (const raw of parts) {
        const part = raw as PartLike;
        if (part.text) {
            try {
                const fromText = await tryTextAsImage(part.text);
                if (fromText) return fromText;
            } catch {
                /* continue */
            }
        }
    }

    return null;
};

const getResponseParts = (response: GenerateContentResponse): PartLike[] | undefined => {
    const c = response.candidates?.[0]?.content;
    const parts = c?.parts as PartLike[] | undefined;
    return parts;
};

/** Text parts that are not duplicate image URLs (API often echoes fileUri in text). */
const extractAuxiliaryText = (parts: PartLike[] | undefined): string | null => {
    if (!parts?.length) return null;
    const urlSet = new Set<string>();
    for (const raw of parts) {
        const p = raw as PartLike;
        const fd = getFileData(p);
        const uri = fd?.fileUri ?? fd?.file_uri;
        if (uri?.trim()) urlSet.add(uri.trim());
    }
    const texts: string[] = [];
    for (const raw of parts) {
        const p = raw as PartLike;
        const t = p.text?.trim();
        if (!t) continue;
        if (urlSet.has(t)) continue;
        if (isProbablyHttpUrl(t)) continue;
        texts.push(t);
    }
    return texts.length ? texts.join('\n') : null;
};

export type ImageEditResult = {
    newBase64Image: string | null;
    text: string | null;
    /** Present when the model returns image bytes or a fetchable URL; use for data: URL mime. */
    outputMimeType?: string;
};

export const generateText = async (prompt: string): Promise<string> => {
    if (!prompt) return "Error: Prompt is empty.";
    try {
        const response: GenerateContentResponse = await generateContentWithRetry({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });
        return response.text;
    } catch (error) {
        return formatError(error, "generateText");
    }
};

export const generateImage = async (prompt: string): Promise<string> => {
    if (!prompt) return "Error: Prompt is empty.";
    try {
        const response: GenerateContentResponse = await generateContentWithRetry({
            model: 'gemini-3-pro-image-preview',
            contents: prompt,
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });

        const parts = getResponseParts(response);
        let img = await extractImageFromGenerateContentParts(parts);
        if (!img && response.text) {
            img = await tryTextAsImage(response.text);
        }
        if (img) {
            return `data:${img.mimeType};base64,${img.base64}`;
        }
        const block = (response as { promptFeedback?: { blockReason?: string } }).promptFeedback;
        if (block?.blockReason) {
            return `Error: Image blocked (${block.blockReason}).`;
        }
        return "Error: Image generation failed to produce an image.";
    } catch (error) {
        return formatError(error, "generateImage");
    }
};

export const editImage = async (
    base64Image: string,
    mimeType: string,
    prompt: string
): Promise<ImageEditResult> => {
    if (!base64Image || !prompt) return { newBase64Image: null, text: "Error: Image or prompt is missing." };
    try {
        const imagePart = base64ToGenerativePart(base64Image, mimeType);
        const textPart = { text: prompt };

        const response: GenerateContentResponse = await generateContentWithRetry({
            model: 'gemini-3-pro-image-preview',
            contents: { parts: [imagePart, textPart] },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });

        const parts = getResponseParts(response);
        const img = await extractImageFromGenerateContentParts(parts);
        if (img) {
            return {
                newBase64Image: img.base64,
                text: extractAuxiliaryText(parts),
                outputMimeType: img.mimeType,
            };
        }
        return { newBase64Image: null, text: "Error: No image in model response." };

    } catch (error) {
        return { newBase64Image: null, text: formatError(error, "editImage") };
    }
};

export const executePreset = async (
    inputs: { data: string; mimeType: string }[],
    prompt: string
): Promise<ImageEditResult> => {
    if (inputs.length === 0 || !prompt) {
        return { newBase64Image: null, text: "Error: Image(s) or prompt is missing." };
    }
    try {
        const imageParts = inputs.map(input => base64ToGenerativePart(input.data, input.mimeType));
        const textPart = { text: prompt };
        const allParts = [...imageParts, textPart];

        const response: GenerateContentResponse = await generateContentWithRetry({
            model: 'gemini-3-pro-image-preview',
            contents: { parts: allParts },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });

        const parts = getResponseParts(response);
        const img = await extractImageFromGenerateContentParts(parts);
        if (img) {
            return {
                newBase64Image: img.base64,
                text: extractAuxiliaryText(parts),
                outputMimeType: img.mimeType,
            };
        }
        return { newBase64Image: null, text: "Error: No image in model response." };

    } catch (error) {
        return { newBase64Image: null, text: formatError(error, "executePreset") };
    }
};

/**
 * 部分 Gemini 兼容中转（new_api 等）在 :predictLongRunning 上会校验 `contents`，
 * 而 @google/genai 默认只发 `instances`/`parameters`。通过 SDK 的 extraBody 合并进请求体。
 * 直连 generativelanguage.googleapis.com 时不要设置 GEMINI_API_BASE_URL，避免多余字段。
 */
/** 下载 Gemini 返回的视频 URI（可带 ?key= 或单独 Header）。 */
const fetchVideoBlob = async (uri: string): Promise<Blob> => {
    const key = API_KEY || '';
    const sep = uri.includes('?') ? '&' : '?';
    const withKey = key && !uri.includes('key=') ? `${uri}${sep}key=${encodeURIComponent(key)}` : uri;
    let res = await fetch(withKey, { mode: 'cors', credentials: 'omit' });
    if (!res.ok && key) {
        res = await fetch(uri, {
            mode: 'cors',
            credentials: 'omit',
            headers: { 'x-goog-api-key': key },
        });
    }
    if (!res.ok) {
        throw new Error(`Video download failed (${res.status})`);
    }
    return res.blob();
};

const buildVideoProxyExtraBody = (
    prompt: string,
    base64Image: string | null,
    mimeType: string | null,
): Record<string, unknown> => {
    const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [{ text: prompt }];
    if (base64Image && mimeType) {
        parts.push({ inlineData: { mimeType, data: base64Image } });
    }
    return {
        contents: [{ role: 'user', parts }],
    };
};

export const generateVideo = async (
    base64Image: string | null,
    mimeType: string | null,
    prompt: string,
    onProgress: (message: string) => void,
): Promise<string> => {
     if (!prompt) return "Error: Prompt is empty.";
    try {
        onProgress("Starting video generation...");
        const useProxyContents = Boolean(API_BASE_URL?.trim());

        const config: {
            numberOfVideos: number;
            httpOptions?: { extraBody: Record<string, unknown> };
        } = {
            numberOfVideos: 1,
            ...(useProxyContents
                ? {
                      httpOptions: {
                          extraBody: buildVideoProxyExtraBody(prompt, base64Image, mimeType),
                      },
                  }
                : {}),
        };

        let operation = await generateVideosWithRetry({
            model: 'veo-3.1-generate-preview',
            prompt,
            ...(base64Image && mimeType
                ? {
                      image: {
                          imageBytes: base64Image,
                          mimeType: mimeType,
                      },
                  }
                : {}),
            config,
        });

        onProgress("Video processing has started. This may take a few minutes...");
        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 10000));
            onProgress("Checking video status...");
            operation = await getVideosOperationWithRetry({ operation: operation });
        }

        onProgress("Video processing complete. Fetching video...");
        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;

        if (!downloadLink) {
            throw new Error("Video URI not found in response.");
        }

        const videoBlob = await fetchVideoBlob(downloadLink);
        
        onProgress("Video fetched successfully.");
        return URL.createObjectURL(videoBlob);
    } catch (error) {
        const errorMessage = formatError(error, "generateVideo");
        onProgress(errorMessage);
        return errorMessage;
    }
};

export const utils = {
    fileToGenerativePart,
    base64ToGenerativePart,
};