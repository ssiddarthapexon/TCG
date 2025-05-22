from phi.agent import Agent, RunResponse
from phi.model.openrouter import OpenRouter
import os
from phi.model.ollama import Ollama



user_story = """
Live Chat

Entering a Live Chat in a Community

· Visibility: The "Live Chat" button should be clearly visible at the top of the community page.
· Quick Access: Tapping the "Live Chat" button should immediately open the live chat interface without delays.
· Real-Time Communication: The chat should allow for real-time discussions with other users in the community.

Chat Features:

· Photos: User can attach a photo to a live chat from their camera or their photo library, and preview before sending 
· GIF’s: Users search the Giphy lirbary, attach a GIF to a live chat, and preview before sending. The Gif should be animated upon sending.
· Paid messages: Users can send a private message with a payment request, requiring that the payment is made before the message can be viewed. 
· Reactions: Users can give a thumbs up/down reaction to a specific message by long-pressing it
· Reply to a specific message: Users can reply to a specific message by long-pressing it, and the response will be connected to the original message. 

Live Chat Administration

Hide or Enable the "Live Chat" Button 

· Toggle Functionality: The app should provide a clear and accessible option for hiding or enabling the "live chat" button.
· Immediate Application: Changes made by the administrator should take effect immediately, with the live chat button being either removed or displayed based on the admin's selection.
· Visibility of Changes: The platform should clearly display the current status of the live chat feature.
· Reversibility: The Community administrator should be able to switch the live chat button on and off as needed.

Pinning a Message in a Live Chat

· Pinning Functionality: The chat interface should provide an option for admins to pin and unpin messages easily.
· Visibility: Pinned messages should remain at the top of the chat window and be visible to all participants, regardless of new messages being sent.
· Admin Control: Only admins should have the ability to pin and unpin messages, ensuring that the feature is used to highlight important information.
· Multiple Pinned Messages: Consider allowing multiple messages to be pinned, or provide a clear indication if only one message can be pinned at a time, with the option to replace the currently pinned message.
· Unpinning: Admins should be able to unpin a message at any time, returning it to its original place in the chat history or removing it from the pinned position
"""

# Test Case Generation Agent (One Intelligent Agent for Multiple Tasks)
test_case_agent = Agent(
    name="Test Case Generation Agent",
    role="Analyze user stories, generate test case titles, and expand them into detailed test steps.",
    #model=OpenRouter(
    #    model="nvidia/llama-3.3-nemotron-super-49b-v1:free",
    #    api_key="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"
    #),
    model=Ollama(id="llama3.1"),
    instructions=[
    "You are a Test Case Generator for a software application.",
    "Given a user story or feature description, you will:",
    "- Analyze the user story to identify relevant scenarios.",
    "- Generate clear and descriptive test case titles for each scenario.",
    "- Expand each test case title into detailed test steps, including:",
    "  - Test Type (Positive, Negative, Edge, Security, Performance, UI/UX)",
    "  - Priority (High, Medium, Low) based on the importance of the test.",
    "  - Preconditions",
    "  - Steps to execute (in bullet points)",
    "  - Expected results",
    "- Always follow this structured format for each test case:",
    "",
    "Test Case Title: <Descriptive Title>",
    "Test Type: <Positive | Negative | Edge | Security | Performance | UI/UX>",
    "Priority: <High | Medium | Low>",
    "Preconditions:",
    "- <List of preconditions>",
    "Steps:",
    "- Step 1: <Detailed step description>",
    "- Step 2: <Detailed step description>",
    "- Step 3: <Detailed step description>",
    "Expected Result:",
    "- <Clear and specific expected outcome>",
    "",
    "- Cover all test types:",
    "  - Positive Scenarios (Valid actions)",
    "  - Negative Scenarios (Invalid actions)",
    "  - Edge Scenarios (Boundary conditions)",
    "  - Security Scenarios (Unauthorized access, data protection)",
    "  - Performance Scenarios (Speed, load handling)",
    "  - UI/UX Scenarios (Visual and interaction tests)",
    ],
    markdown=True,
    structured_outputs=True
)

# Validation Agent
validation_agent = Agent(
    name="Validation Agent",
    role="Validate test cases for completeness, accuracy, and clarity.",
    model=Ollama(id="llama3.1"),
    instructions=[
        "You are a Test Case Validator.",
        "Given a test case, you will:",
        "- Check if the test case title is clear and descriptive.",
        "- Ensure the steps are complete, accurate, and follow a logical flow.",
        "- Validate the expected result for accuracy.",
        "- Suggest improvements if any issues are found."
    ],
    markdown=True,
    structured_outputs=True
)

# Optimization Agent
optimization_agent = Agent(
    name="Optimization Agent",
    role="Optimize test cases by removing redundant cases and ensuring coverage.",
    model=Ollama(id="llama3.1"),
    instructions=[
        "You are a Test Case Optimization Agent.",
        "Given a list of test cases, you will:",
        "- Identify and remove any duplicate or redundant test cases.",
        "- Merge similar test cases for better coverage.",
        "- Ensure all necessary scenarios are covered (positive, negative, edge)."
    ],
    markdown=True,
    structured_outputs=True
)


test_case_team = Agent(
    team=[test_case_agent, validation_agent, optimization_agent],
    model=Ollama(id="llama3.1"),
    instructions=[
        "Analyze user stories, generate test cases, validate them, and optimize them for clarity and coverage.",
        "Always ensure clear, complete, and optimized test cases."
    ],
    markdown=True,
    structured_outputs=True
)

# Test the Agent Team
from pprint import pprint

test_case_team.print_response(
    f"Generate and validate test cases for the following user story:\n{user_story}",
    stream=True
)
