# app/schemas.py
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import Optional, Any, List

class IngestText(BaseModel):
    name: str
    text: str
    reference_time: Optional[str] = None
    source_description: Optional[str] = "app"
    group_id: Optional[str] = None

class IngestMessage(BaseModel):
    name: str
    messages: List[str]
    reference_time: Optional[str] = None
    source_description: Optional[str] = "chat"
    group_id: Optional[str] = None

class IngestJSON(BaseModel):
    name: str
    data: Any = Field(alias="json")  # tránh trùng tên method .json()
    reference_time: Optional[str] = None
    source_description: Optional[str] = "json"
    group_id: Optional[str] = None

    # Pydantic v2 config: allow population by field alias ("json" -> data)
    model_config = ConfigDict(populate_by_name=True)


class SearchRequest(BaseModel):
    query: str
    focal_node_uuid: Optional[str] = None
    group_id: Optional[str] = None  # Filter by conversation group
