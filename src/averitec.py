from dataclasses import dataclass


@dataclass
class Datapoint:
    """
    A dataclass representing a claim and its associated metadata.

    Attributes:
        claim (str): The text of the claim.
        claim_id (int): The unique identifier for the claim.
        claim_date (str, optional): The date the claim was made.
        speaker (str, optional): The person or entity who made the claim.
        original_claim_url (str, optional): URL to the original claim.
        reporting_source (str, optional): The source reporting the claim.
        location_ISO_code (str, optional): ISO code for the location relevant to the claim.
        label (str, optional): The veracity label of the claim.
        metadata (dict, optional): Additional metadata associated with the claim.
    """

    claim: str
    claim_id: int
    claim_date: str = None
    claim_images: list = None
    speaker: str = None
    original_claim_url: str = None
    reporting_source: str = None
    location_ISO_code: str = None
    label: str = None
    metadata: dict = None
    justification: str = None

    @classmethod
    def from_dict(cls, json_data: dict, claim_id: int = None):
        json_data = json_data.copy()
        metadata = json_data.pop("metadata", {})
        return cls(
            claim=json_data.pop("claim_text"),
            claim_id=json_data.pop("claim_id", claim_id),
            claim_date=json_data.pop("date", None),
            claim_images=json_data.pop("claim_images", None),
            speaker=metadata.pop("speaker", None),
            original_claim_url=metadata.pop("original_claim_url", None),
            reporting_source=metadata.pop("reporting_source", None),
            location_ISO_code=json_data.pop("location", None),
            label=json_data.pop("label", None),
            justification=json_data.pop("justification", None),
            metadata=metadata,
        )

    def to_dict(self):
        metadata = self.metadata.copy() if self.metadata is not None else {}
        metadata["speaker"] = self.speaker
        metadata["original_claim_url"] = self.original_claim_url
        metadata["reporting_source"] = self.reporting_source
        return {
            "claim_text": self.claim,
            "claim_id": self.claim_id,
            "date": self.claim_date,
            "claim_images": self.claim_images,
            "speaker": self.speaker,
            "metadata": metadata,
            "location": self.location_ISO_code,
            "label": self.label,
            "justification": self.justification,
        }
