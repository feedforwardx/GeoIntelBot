from typing import List
from pydantic import BaseModel, Field

class AtomicFact(BaseModel):
    
    key_elements: List[str] = Field(description="""The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the atomic fact's narrative.""")
    
    atomic_fact: str = Field(description="""The smallest, indivisible facts, presented as concise sentences. These include
propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.""")


class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(description="List of atomic facts")