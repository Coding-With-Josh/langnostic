import Image from "next/image";
import { Highlighter } from "../ui/highlighter";

/**
 * Submission-state hero: the site is the front door to the finished study, so
 * it reads as the pitch (what was measured, what changed), not as a status
 * report on how far the work has got. `data-reveal` lives on the wrapper in
 * `page.tsx`, which owns the page-load gate.
 */
const Hero = () => {
  return (
    <div className="rise flex flex-col items-center gap-8 px-6 pt-24 pb-28 text-center sm:pt-32">
      <Image
        src="/langnostic.svg"
        alt="langnostic"
        width={520}
        height={144}
        loading="eager"
        fetchPriority="high"
        unoptimized
        className="h-14 w-auto"
      />

      <div className="flex max-w-[50rem] flex-col items-center gap-4">
        <h1 className="max-w-[26ch] text-balance text-4xl leading-[1.05] tracking-[-0.03em] sm:text-5xl">
          a study of{" "}
          <Highlighter action="circle" color="#FF9D00">
            {" "}
            whisper-small{" "}
          </Highlighter>{" "}
          across languages and accents
          {/* a study of <span className="font-thin italic">whisper-small</span> across languages and accents */}
        </h1>

        <p className="max-w-3xl text-balance font-sans text-md leading-relaxed tracking-[-0.01em] text-black/60">
          i measure <span className="italic">whisper-small</span> across
          languages and accents, and test whether a small lora adapter trained
          on 30 minutes of audio closes the gap.
        </p>
      </div>

      <div className="flex items-center justify-center gap-3">
        <a
          href="#results"
          className="flex h-12 min-w-fit items-center justify-center rounded-full bg-black px-5 font-sans text-md font-medium tracking-normal text-white transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          see results
        </a>
        <a
          href="#demo"
          className="flex h-12 min-w-fit items-center justify-center rounded-full bg-black/10 px-6 font-sans text-md font-medium tracking-normal text-black transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          watch demo
        </a>
      </div>
    </div>
  );
};

export default Hero;
