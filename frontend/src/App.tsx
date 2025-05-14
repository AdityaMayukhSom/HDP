function App() {
  const projectTitle =
    "Abstractive Summarization with PG-LLaMA and Evaluation of Hallucinations using Token Probability and EntFA";
  return (
    <main className="p-8">
      <h1 className="text-xl mb-8">{projectTitle}</h1>
      <form action="" className="flex flex-col w-full gap-y-2">
        <label htmlFor="abstract-text">
          Enter paper abstract of link here:
        </label>
        <textarea id="abstract-text" className="border border-black"></textarea>
        <button type="submit" className="bg-emerald-300 p-2">
          Summarize
        </button>
      </form>
      <form>
        <p></p>
        <button type="submit"></button>
      </form>

      <div></div>
    </main>
  );
}

export default App;
