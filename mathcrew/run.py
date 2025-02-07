from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
import os
import asyncio
import re
from dotenv import load_dotenv
from naptha_sdk.schemas import AgentRunInput, NodeConfigUser, AgentConfig, LLMConfig
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage

logger = get_logger(__name__)
load_dotenv()

# Define a Pydantic model for the calculator input.
class CalculatorInput(BaseModel):
    expression: str

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            args_type=CalculatorInput,  # Use a Pydantic model to ensure schema compatibility.
            return_type=float,
            name="MathCalculator",
            description="Performs mathematical calculations with input validation"
        )
    
    async def run(self, input_data: CalculatorInput, cancellation_token: CancellationToken = None) -> float:
        expression = input_data.expression
        
        # Remove a leading "calculate" keyword if present.
        if expression.strip().lower().startswith("calculate"):
            expression = expression.strip()[len("calculate"):].strip()
        
        # Replace percentage expressions like "15% of 300" or "15% * 300"
        # with a valid arithmetic expression: "(15/100)*300".
        expression = re.sub(r'(\d+)%\s*(?:of|\*)\s*(\d+)', r'(\1/100)*\2', expression)
        
        # Clean up whitespace.
        expression = expression.strip()
        
        # Validate allowed characters. After substitution, allowed characters: digits, operators, dot, parentheses.
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression.replace(" ", "")):
            raise ValueError(f"Invalid characters in expression: {expression}")
        
        # Evaluate the arithmetic expression.
        # We assume the expression is safe because of the allowed-character check.
        return eval(expression)

class MathCrew:
    def __init__(self):
        self.tool = CalculatorTool()
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Create the assistant agent with the calculator tool.
        self.assistant = AssistantAgent(
            name="MathExpert",
            system_message="Use calculator for math problems. Reply TERMINATE when done.",
            model_client=model_client,
            tools=[self.tool]
        )

    async def solve(self, problem: str) -> str:
        # Construct a conversation history with one user message.
        messages = [TextMessage(content=problem, source="user")]
        cancellation_token = CancellationToken()
        # Use the assistant's on_messages method to generate a reply.
        response = await self.assistant.on_messages(messages, cancellation_token)
        # Extract and return the assistant's reply content.
        return response.chat_message.content

def run(module_run: dict) -> dict:
    validated = AgentRunInput(**module_run)
    if validated.inputs.get('func_name') != "mathcrew":
        raise ValueError("Only mathcrew function supported")
    
    math_crew = MathCrew()
    problem = validated.inputs.get('func_input_data', {}).get("description", "")
    if not problem:
        return {"error": "No problem in description"}
    
    result = asyncio.run(math_crew.solve(problem))
    return {"result": result}

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()
    deployment = asyncio.run(
        setup_module_deployment(
            "agent",
            "mathcrew/configs/deployment.json",
            node_url=os.getenv("NODE_URL"),
            user_id=None,
            load_persona_data=False,
            is_subdeployment=False
        )
    )

    example_inputs = {
        "description": "Calculate (15% of 300) + (25% of 800)",
        "expected_output": "245"
    }

    module_run = {
        "inputs": {
            "func_name": "mathcrew",
            "func_input_data": example_inputs
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = run(module_run)
    print(response)
