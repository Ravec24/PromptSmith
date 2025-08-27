# promptsmith_agent.py
from __future__ import annotations

from typing import Literal, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import sys


# ---------- Pydantic models (optional; kept for structure clarity) ----------

class PromptRequest(BaseModel):
    """Model for prompt improvement requests."""
    content: str = Field(description="The original prompt or idea to improve")
    domain: Literal["text", "code", "image"] = Field(description="Target domain for the prompt")
    context: str = Field(default="", description="Additional context or requirements")


class AdditionalDataRequest(BaseModel):
    """Model for requesting additional data from user."""
    question: str = Field(description="Specific question to ask the user")
    reason: str = Field(description="Why this information is needed")


# ---------- Tools ----------

@tool
def request_additional_data(query: str, reason: str) -> str:
    """
    Ask the user for more specific information to improve the prompt.

    Args:
        query: The specific question to ask the user.
        reason: Explanation of why this information is needed.

    Returns:
        A formatted message indicating that user input is requested.
    """
    return f"""
🔍 **Need More Information**

**Question:** {query}

**Why we need this:** {reason}

Please provide this information so I can create a better prompt.
""".strip()


@tool
def create_text_prompt(
    original_content: str,
    purpose: str,
    tone: str = "professional",
    audience: str = "general",
    length: str = "medium",
) -> str:
    """
    Create an improved prompt for text generation tasks.

    Args:
        original_content: The original prompt or idea.
        purpose: What the text should accomplish.
        tone: Desired tone (professional, casual, creative, etc.).
        audience: Target audience.
        length: Desired length (short, medium, long).

    Returns:
        An improved text generation prompt.
    """
    improved_prompt = f"""
📝 **IMPROVED TEXT PROMPT**

**Role & Context:** You are an expert writer specializing in {purpose}.

**Task:** {original_content}

**Requirements:**
- Tone: {tone}
- Target Audience: {audience}
- Length: {length}
- Structure: Clear introduction, well-developed body, strong conclusion

**Quality Guidelines:**
- Use engaging and appropriate language for {audience}
- Maintain {tone} tone throughout
- Include specific examples where relevant
- Ensure logical flow and coherence
- End with a compelling conclusion

**Format:** Provide the content in a clean, readable format suitable for the intended {audience}.
""".strip()
    return improved_prompt


@tool
def create_code_prompt(
    original_content: str,
    programming_language: str,
    complexity: str = "intermediate",
    include_tests: bool = True,
    include_comments: bool = True,
) -> str:
    """
    Create an improved prompt for code generation tasks.

    Args:
        original_content: The original coding request.
        programming_language: Target programming language.
        complexity: Code complexity level (beginner, intermediate, advanced).
        include_tests: Whether to include test cases.
        include_comments: Whether to include detailed comments.

    Returns:
        An improved code generation prompt.
    """
    test_section = (
        "- Include comprehensive test cases\n- Add error handling examples"
        if include_tests
        else "- Focus on core implementation"
    )

    comment_section = (
        "- Add detailed inline comments explaining logic\n- Include docstrings for functions/classes"
        if include_comments
        else "- Keep comments minimal and focused"
    )

    deliverable_tests = "4. Test cases with expected outputs" if include_tests else ""

    improved_prompt = f"""
💻 **IMPROVED CODING PROMPT**

**Role:** You are a senior {programming_language} developer with expertise in clean code principles.

**Task:** {original_content}

**Technical Requirements:**
- Language: {programming_language}
- Complexity Level: {complexity}
- Follow best practices and design patterns
{test_section}
{comment_section}

**Code Quality Standards:**
- Use meaningful variable and function names
- Implement proper error handling
- Follow language-specific conventions
- Optimize for readability and maintainability
- Include type hints where applicable

**Deliverables:**
1. Clean, working code
2. Brief explanation of approach
3. Usage examples
{deliverable_tests}

**Format:** Provide code in properly formatted blocks with syntax highlighting.
""".strip()
    return improved_prompt


@tool
def create_image_prompt(
    original_content: str,
    style: str = "realistic",
    composition: str = "centered",
    lighting: str = "natural",
    color_scheme: str = "vibrant",
    quality_level: str = "high",
) -> str:
    """
    Create an improved prompt for image generation tasks.

    Args:
        original_content: The original image idea.
        style: Visual style (realistic, artistic, cartoon, cyberpunk, fantasy, etc.).
        composition: Image composition (centered, rule of thirds, etc.).
        lighting: Lighting type (natural, dramatic, soft, etc.).
        color_scheme: Color palette (vibrant, muted, monochrome, etc.).
        quality_level: Quality indicators (high, ultra-high, professional).

    Returns:
        An improved image generation prompt.
    """
    style_enhancements = {
        "realistic": "photorealistic, highly detailed",
        "artistic": "artistic masterpiece, creative interpretation",
        "cartoon": "stylized cartoon, clean lines",
        "cyberpunk": "cyberpunk aesthetic, neon lights, futuristic",
        "fantasy": "fantasy art, magical elements",
    }

    improved_prompt = f"""
🎨 **IMPROVED IMAGE PROMPT**

**Main Subject:** {original_content}

**Visual Style:** {style_enhancements.get(style, style)}, {quality_level} quality

**Composition & Framing:**
- Layout: {composition}
- Perspective: Engaging viewpoint
- Focus: Sharp subject with appropriate depth of field

**Lighting & Atmosphere:**
- Lighting: {lighting} lighting
- Mood: Captivating and immersive
- Atmosphere: Professional photography quality

**Color & Aesthetics:**
- Color Scheme: {color_scheme} colors
- Contrast: Well-balanced highlights and shadows
- Saturation: Optimized for visual impact

**Technical Quality:**
- Resolution: {quality_level} resolution
- Details: Crisp, sharp details
- Rendering: Professional-grade output

**Additional Enhancements:**
- Background: Complementary and non-distracting
- Textures: Realistic surface details
- Overall: Visually striking and memorable

**Negative Prompts:** blurry, low quality, distorted, oversaturated, cluttered background
""".strip()
    return improved_prompt


# ---------- Agent ----------

class PromptSmithAgent:
    """
    PromptSmith agent that improves prompts for text, code, or image generation
    using a prebuilt ReAct agent and four tools.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        # Load environment variables (e.g., from .env)
        load_dotenv()

        # Configure model and credentials
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Set it in your environment or .env file."
            )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=temperature,
            google_api_key=self.api_key,
        )

        # Register tools
        self.tools = [
            request_additional_data,
            create_text_prompt,
            create_code_prompt,
            create_image_prompt,
        ]

        # Build agent with a robust prompt parameter fallback for different versions
        sys_prompt = self._get_system_prompt()

        # Primary: messages_modifier; Fallback: prompt
        try:
            self.agent = create_react_agent(
                self.llm,
                self.tools,
                messages_modifier=sys_prompt,
            )
        except TypeError:
            # Older/newer variants may expose "prompt" instead
            self.agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=sys_prompt,
            )

    @staticmethod
    def _get_system_prompt() -> str:
        return """You are PromptSmith, an expert prompt engineering agent. Improve prompts for text, code, or image generation.

Process:
1. Analyze the user's request to identify the target domain (text, code, or image).
2. If information is insufficient, call the `request_additional_data` tool with a specific, concise question.
3. Once sufficient info is gathered, call exactly one of:
   - `create_text_prompt` for text generation
   - `create_code_prompt` for programming tasks
   - `create_image_prompt` for image generation

Guidelines:
- Ask for clarification when the request is vague or missing key constraints.
- Output detailed, actionable prompts that drive high‑quality results.
- Tailor constraints (tone, audience, complexity, style) to the request and any provided context.
"""

    def improve_prompt(self, user_request: str) -> str:
        """
        Improve the user's prompt via the agent orchestration.

        Args:
            user_request: The raw prompt or request from the user.

        Returns:
            The improved prompt (or a clarification question request).
        """
        result = self.agent.invoke({"messages": [HumanMessage(content=user_request)]})
        # Extract the final content safely
        try:
            messages = result.get("messages") if isinstance(result, dict) else None
            if messages:
                last = messages[-1]
                # Some message types store text in .content
                return getattr(last, "content", str(last))
            return str(result)
        except Exception:
            return str(result)


# ---------- CLI demo ----------

def main() -> None:
    print("🚀 PromptSmith Agent Ready!")
    print("-" * 60)

    # Allow override via CLI: python promptsmith_agent.py "your request"
    if len(sys.argv) > 1:
        user_req = " ".join(sys.argv[1:])
        agent = PromptSmithAgent()
        print("\nUser Request:")
        print(user_req)
        print("\nImproved Prompt / Follow-up:")
        print(agent.improve_prompt(user_req))
        return

    # Otherwise run sample batch
    agent = PromptSmithAgent()
    example_requests = [
        "Help me create a prompt for writing a blog post about AI trends in 2025 for non-technical readers.",
        "I need to generate Python code for a web scraper that extracts article titles from a news site.",
        "Create an image of a futuristic cityscape with flying cars and neon lighting in a cyberpunk style.",
        "Write a detailed marketing email, but I didn't provide target audience or tone.",
    ]

    for i, req in enumerate(example_requests, 1):
        print(f"\nExample {i}: {req}")
        print("-" * 30)
        print(agent.improve_prompt(req))
        print("=" * 60)


if __name__ == "__main__":
    main()
