import axios, { type AxiosResponse } from "axios";
import { Config } from "./config";

import {
  useState,
  useRef,
  type FormEventHandler,
  type MouseEventHandler,
} from "react";
import {
  Apis,
  SummarizeResponseSchema,
  type SummarizeResponse,
  type SummarizeRequest,
  type ExtractFeaturesResponse,
  type ExtractFeaturesRequest,
} from "./apis";

function App() {
  const [highlight, setHighlight] = useState<string>("");
  const [abstract, setAbstract] = useState<string>("");

  const handleCheckForHallucination: MouseEventHandler<
    HTMLButtonElement
  > = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const res = await axios.post<
      ExtractFeaturesResponse,
      AxiosResponse<ExtractFeaturesResponse, ExtractFeaturesRequest>,
      ExtractFeaturesRequest
    >(
      Apis.EXTRACT_FEATURES,
      {
        abstract: abstract,
        highlight: highlight,
      } //
    );

    console.log(res.data);
  };

  const handleSummarizeFormSubmit: FormEventHandler<
    HTMLFormElement //
  > = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    try {
      const res = await axios.post<
        SummarizeResponse,
        AxiosResponse<SummarizeResponse, SummarizeRequest>,
        SummarizeRequest
      >(
        Apis.SUMMARIZE,
        {
          abstract: abstract,
        } //
      );

      const data = SummarizeResponseSchema.parse(res.data);

      setHighlight(data.highlight);
    } catch (e) {
      console.error(e);
      setHighlight("");
    }
  };

  return (
    <main className="px-8 py-4 min-h-svh w-svw bg-background text-text">
      <h1 className="text-xl mb-4 underline underline-offset-4">
        {Config.PROJECT_TITLE}
      </h1>
      <section className="grid grid-cols-2 gap-x-8">
        <section className="">
          <form
            onSubmit={handleSummarizeFormSubmit}
            className="flex flex-col w-full gap-y-2"
          >
            <label htmlFor="abstract" className="text-lg">
              Enter paper abstract or science direct link here:
            </label>
            <textarea
              id="abstract"
              name="abstract"
              className="border px-3 py-2 field-sizing-content max-h-96"
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
            ></textarea>
            <button
              type="submit"
              // disabled={status.pending}
              className="bg-secondary p-2 cursor-pointer"
            >
              Summarize
            </button>
          </form>
        </section>
        <section>
          {highlight && (
            <section className="flex flex-col">
              <div className="mb-6">
                <h1 className="py-2 text-lg">Model Generated Highlight: </h1>
                <div>{highlight}</div>
              </div>
              <button
                type="button"
                onClick={handleCheckForHallucination}
                className="bg-secondary py-2 px-8 cursor-pointer self-end-safe"
              >
                Check For Hallucination
              </button>
            </section>
          )}
        </section>
      </section>
    </main>
  );
}

export default App;
