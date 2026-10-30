import "./index.css";
import MaterialsCloudHeader from "mc-react-header";
import { useMemo, useState } from "react";

import InfoAccordions from "./AccordionContent";
import AlphaML from "./AlphaML";
import { toAlphaMLShare } from "./AlphaML/share";
import MoleculeUpload from "./FileUpload";
import { PAPER_DOI } from "./utils";

import { McAppHeader, McShareButton } from "mc-react-library";

// kv-store for share links. .env.development points `pnpm dev` at the dev
// store; unset (production builds) means the Materials Cloud production store.
const KV_URL = import.meta.env.VITE_KV_URL;

function App() {
  // { structure, fileName, format, shareId? }
  const [loaded, setLoaded] = useState(null);
  // The result shown, { model, data, fromLink }: shared with the structure,
  // and only once there is one (no Share button before)
  const [result, setResult] = useState(null);
  const shared = useMemo(() => loaded && result && { ...loaded, result }, [loaded, result]);

  return (
    <>
      <MaterialsCloudHeader
        activeSection="work"
        breadcrumbsPath={[
          { name: "Work", link: "https://www.materialscloud.org/work" },
          { name: "Tools", link: "https://www.materialscloud.org/work/tools" },
          { name: "AlphaML", link: null },
        ]}
      />

      <main className="min-h-screen bg-[aliceblue] pb-10">
        <div className="mx-auto max-w-7xl px-4">
          <div className="overflow-hidden bg-white p-4 shadow-sm">
            <McAppHeader
              title={
                <>
                  <span className="font-bold">AlphaML</span>: machine learning
                  of molecular polarizabilities
                </>
              }
              logo={`${import.meta.env.BASE_URL}alphaml.png`}
              logoAlt="AlphaML logo"
              dois={[{ doi: PAPER_DOI, label: "PNAS" }]}
            />
            <div className="mx-auto max-w-4xl px-6 py-8 sm:px-10">
              <div className="space-y-3">
                <InfoAccordions />
                <MoleculeUpload kvUrl={KV_URL} onStructureParsed={setLoaded} />
              </div>
            </div>

            <AlphaML
              structure={loaded?.structure}
              fileName={loaded?.fileName}
              shareId={loaded?.shareId}
              kvUrl={KV_URL}
              onResult={setResult}
            />

            <McShareButton
              kvUrl={KV_URL}
              data={shared}
              toDocument={toAlphaMLShare}
              // The result came with the link opened: share that link again
              shareId={result?.fromLink ? loaded?.shareId : undefined}
              className="mx-auto max-w-6xl px-6 sm:px-10"
            />
          </div>
        </div>
      </main>
    </>
  );
}

export default App;
