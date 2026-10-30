// Example molecules loaded on click from /public/examples (the set of the
// legacy AlphaML Flask app, webservice/file_examples).
export const examples = [
  { label: "Caffeine", file: "Caffeine.xyz" },
  { label: "Aciclovir", file: "Aciclovir.xyz" },
  { label: "Cytosine", file: "Cytosine.xyz" },
  { label: "Methionine", file: "Methionine.xyz" },
  { label: "Fullerene", file: "Fullerene.xyz" },
  { label: "Pentacene", file: "Pentacene.xyz" },
].map((example) => ({ ...example, url: `/examples/${example.file}` }));
