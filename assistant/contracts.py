from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Reply(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['answer','clarify','refuse','conflict']
    answer:str=Field(min_length=1,max_length=1200)


class Citation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:str=Field(min_length=1,max_length=40)
    quote:str=Field(min_length=1,max_length=280)


class GroundedReply(Reply):
    evidence:list[Citation]=Field(max_length=3)
    conflict_reason:str=Field(default='',max_length=300)


class Rewrite(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['search','clarify']
    query:str=Field(max_length=900)
    clarification:str=Field(max_length=350)
    @model_validator(mode='after')
    def validate_action(self):
        if self.status=='search' and not self.query.strip():
            raise ValueError('Поисковый запрос пуст')
        if self.status=='clarify' and not self.clarification.strip():
            raise ValueError('Уточняющий вопрос пуст')
        return self


class StandaloneQuery(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query:str=Field(max_length=900)
    clarification:str=Field(default='',max_length=350)
    @model_validator(mode='after')
    def has_action(self):
        if not self.query.strip() and not self.clarification.strip():raise ValueError('Не задан вопрос или уточнение')
        return self


def grounded_schema(aliases):
    schema=GroundedReply.model_json_schema()
    schema['$defs']['Citation']['properties']['id']={'type':'string','enum':list(aliases)}
    return schema
