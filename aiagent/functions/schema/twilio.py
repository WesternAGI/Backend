from typing import Dict, Any
from pydantic import BaseModel, Field

# Use the shared function schema decorator (compat shim re-exports correct impl)
from server.utils.functions_metadata import function_schema

# We delegate to the backend service that already wraps Twilio
from server.services import send_twilio_message as _send_twilio_message


class TwilioSendMessageResponse(BaseModel):
    message_sid: str = Field(..., description="Twilio message SID")
    status: str | None = Field(None, description="Delivery status returned by Twilio")
    to: str | None = Field(None, description="Recipient phone number")
    date_created: str | None = Field(None, description="Creation timestamp from Twilio")


@function_schema(
    name="send_twilio_message",
    description=(
        "Send an SMS via Twilio to a target phone number. "
        "Use E.164 format for the phone number (e.g., +14155552671)."
    ),
    required_params=["to_phone_number", "message"],
)
def send_twilio_message(to_phone_number: str, message: str) -> Dict[str, Any]:
    """
    AI tool: Send an SMS using the server's Twilio integration.

    Args:
        to_phone_number: Recipient phone number in E.164 format (e.g., +14155552671)
        message: Text message content

    Returns:
        A dictionary with Twilio message metadata (SID, status, etc.)
    """
    # Delegate to the server service which handles auth, errors, and config
    result = _send_twilio_message(to_phone_number=to_phone_number, message=message)

    # Ensure the response is JSON-serialisable and matches our model shape
    return {
        "message_sid": result.get("message_sid"),
        "status": result.get("status"),
        "to": result.get("to"),
        "date_created": result.get("date_created"),
    }
