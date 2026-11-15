import { LoadingCounter } from "@/components/loading-counter";
import Hero from "@/components/section/hero";
import Report from "@/components/section/report";

export default function Home() {
  return (
    <div className="relative flex flex-col items-center justify-center bg-white text-black min-h-screen">
      {/* `data-reveal` is the page-load gate: the hero is held back until the
          counter reaches 100%, so the animation is the page load rather than an
          ornament beside it. Rules live in globals.css and fail open. It sits
          on this wrapper (not on `Hero`) so there is exactly one fade. */}
      <div data-reveal>
        <Hero />
         <Report />
      </div>
      {/* `fixed`, not `absolute`: this wrapper's parent is as tall as the whole
          report, so `absolute bottom-0` put the counter at the bottom of the
          *document* — tens of thousands of pixels below the fold, which is why
          nothing ever appeared. `fixed` pins it to the viewport, which is the
          frame the count is about. */}
      <div className="fixed bottom-0 left-4 text-[100px]">
        <LoadingCounter />
      </div>
    </div>
  );
}
