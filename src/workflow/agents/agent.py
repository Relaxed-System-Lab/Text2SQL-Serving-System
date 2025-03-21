from workflow.system_state import SystemState
from workflow.agents.tool import Tool
import re

from llm.models import call_engine, get_llm_chain
from llm.prompts import get_prompt

class Agent:
    """
    Abstract base class for agents.
    """
    
    def __init__(self, name: str, task: str, config: dict):
        self.name = name
        self.task = task
        self.config = config
        self.tools_config = config["tools"]
        self.tools = {}
        self.chat_history = []
    
    def workout(self, system_state: SystemState) -> SystemState:
        """
        Abstract method to process the system state.

        Args:
            system_state (SystemState): The current system state.

        Returns:
            SystemState: The processed system state.
        """
        if self.name == "Candidate Generator":
            try:
                tool = self.tools["generate_candidate"]
                try:
                    tool_response = self.call_tool(tool, system_state)
                except Exception as e:
                    print(f"Error in tool {tool.tool_name}: {e}")
                tool = self.tools["revise"]
                for i in range(8):
                    if tool.need_to_fix(system_state):
                        try:
                            tool_response = self.call_tool(tool, system_state)
                        except Exception as e:  
                            print(f"Error in tool {tool.tool_name}: {e}")
                    else:
                        print(f"Agent {self.name} done in {i}", flush=True)
                        break
            except Exception as e:
                print(f"Error in agent {self.name}: {e}")
            return system_state
        else:
            system_prompt = get_prompt(template_name="agent_prompt")
            system_prompt = system_prompt.format(agent_name=self.name, 
                                                task=self.task, 
                                                tools=self.get_tools_description())
            try:
                for i in range(8):
                    response = self.call_agent(system_prompt, system_state)
                    print(f"Agent {self.name} response: {response}", flush=True)
                    if self.is_done(response):
                        print(f"Agent {self.name} done in {i}", flush=True)
                        break
                    tool_name = self.get_next_tool_name(response)
                    tool = self.tools[tool_name]
                    try:
                        tool_response = self.call_tool(tool, system_state)
                        self.chat_history.append(tool_name)
                    except Exception as e:
                        print(f"Error in tool {tool_name}: {e}")
            except Exception as e:
                print(f"Error in agent {self.name}: {e}")
                
            return system_state

    def call_tool(self, tool: Tool, system_state: SystemState) -> SystemState:
        """
        Call a tool with the given name and system state.
        """
        try:
            tool(system_state)
            return f"Tool {tool.tool_name} called successfully. What's next?"
        except Exception as e:
            raise e
        
    def is_done(self, response: str) -> bool:
        """
        Check if the response indicates that the agent is done.
        """
        return "DONE" in response
    
    def get_next_tool_name(self, response: str) -> str:
        """
        Get the next tool to call based on the response.
        """
        tool_name = response.split("ANSWER:")[1].strip()
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return tool_name
    
    def call_agent(self, system_prompt, system_state: SystemState) -> SystemState:
        """
        Call the agent with the given system state.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"The following tools have been called in order: \n{str(self.chat_history)}\n"})

        llm_chain = get_llm_chain(engine_name=self.config["engine"], temperature=0)
        response = call_engine(name=self.name, message=messages, engine=llm_chain)
        return response
        
    def get_tools_description(self) -> str:
        """
        Get the description of the tools.
        """
        tools_description = ""
        for i, tool in enumerate(self.tools):
            tools_description += f"{i+1}. {tool}\n"
        return tools_description
    
    def __call__(self, system_state: SystemState) -> SystemState:
        return self.workout(system_state)