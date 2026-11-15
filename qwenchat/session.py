from dataclasses import dataclass, field

@dataclass
class GenerationParams:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    show_thinking: bool = True
    max_tool_rounds: int = 5          # safety limit for multi-step tools

    def summary(self) -> str:
        return (
            f"temp={self.temperature}  "
            f"tokens={self.max_tokens}  "
            f"top_p={self.top_p}  "
            f"thinking={'on' if self.show_thinking else 'off'}  "
            f"max_tool_rounds={self.max_tool_rounds}"
        )