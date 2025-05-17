import { z } from "zod";

export const SummarizeRequestSchema = z.object({
  abstract: z.string(),
});
export type SummarizeRequest = z.infer<typeof SummarizeRequestSchema>;

export const SummarizeResponseSchema = z.object({
  highlight: z.string(),
});
export type SummarizeResponse = z.infer<typeof SummarizeResponseSchema>;

export const ExtractFeaturesRequestSchema = z.object({
  abstract: z.string(),
  highlight: z.string(),
});
export type ExtractFeaturesRequest = z.infer<
  typeof ExtractFeaturesRequestSchema
>;

export const ExtractFeaturesResponseSchema = z.object({
  MTP: z.number().finite(),
  AVGTP: z.number().finite(),
  MDVTP: z.number().finite(),
  MMDVP: z.number().finite(),
});
export type ExtractFeaturesResponse = z.infer<
  typeof ExtractFeaturesResponseSchema
>;

export class Apis {
  public static readonly BACKEND_BASE_URI = "";

  public static readonly SUMMARIZE = "/api/summarize";
  public static readonly EXTRACT_FEATURES = "/api/extract/features";
}
