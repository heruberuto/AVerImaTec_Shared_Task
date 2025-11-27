import json

with open("/mnt/data/factcheck/averimatec/val.json", "r") as f:
    data = json.load(f)

claim_id = 15
if claim_id < len(data):
    dp = data[claim_id]
    print(f"Keys in datapoint {claim_id}: {list(dp.keys())}")
    if "claim_images" in dp:
        print(f"Claim images: {dp['claim_images']}")
    if "questions" in dp:
        print(f"Number of questions: {len(dp['questions'])}")
        # Check if questions have images
        for i, q in enumerate(dp["questions"]):
            print(f"Question {i} input_images: {q.get('input_images', [])}")
            # Check answers for source URLs
            for ans in q.get("answers", []):
                print(f"  Answer source: {ans.get('source_url')}")
else:
    print(f"Claim ID {claim_id} out of range (len={len(data)})")
