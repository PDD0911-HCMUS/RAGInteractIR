import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt


class VisDialAnnotation:
    """
    Utility class for reading VisDial annotation JSON files.

    Expected JSON structure:
    {
        "data": {
            "questions": [...],
            "answers": [...],
            "dialogs": [
                {
                    "image_id": ...,
                    "caption": ...,
                    "dialog": [
                        {
                            "question": <int>,
                            "answer": <int>,
                            "answer_options": [<int>, ...],
                            "gt_index": <int>
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        "split": "...",
        "version": "1.0"
    }
    """

    def __init__(self, anno_path: Union[str, Path]) -> None:
        self.anno_path = Path(anno_path)

        if not self.anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.anno_path}")

        with open(self.anno_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        if "data" not in self.raw_data:
            raise ValueError("Invalid VisDial annotation: missing 'data' field.")

        data = self.raw_data["data"]

        self.questions: List[str] = data.get("questions", [])
        self.answers: List[str] = data.get("answers", [])
        self.dialogs: List[Dict[str, Any]] = data.get("dialogs", [])

        self.split: Optional[str] = self.raw_data.get("split")
        self.version: Optional[str] = self.raw_data.get("version")

    def __len__(self) -> int:
        return len(self.dialogs)

    def _validate_dialog_index(self, index: int) -> None:
        if index < 0 or index >= len(self.dialogs):
            raise IndexError(
                f"Dialog index {index} out of range. Valid range: [0, {len(self.dialogs) - 1}]"
            )

    def _resolve_question(self, question_idx: int) -> str:
        return self.questions[question_idx]

    def _resolve_answer(self, answer_idx: int) -> str:
        return self.answers[answer_idx]

    def _resolve_answer_options(self, answer_option_indices: List[int]) -> List[str]:
        return [self.answers[idx] for idx in answer_option_indices]

    def get_dialog_item(self, index: int) -> Dict[str, Any]:
        """
        Return the raw dialog item from data['dialogs'][index].
        """
        self._validate_dialog_index(index)
        return self.dialogs[index]

    def get_script(
        self,
        index: int,
        include_answer_options: bool = False,
        answer_options_as_text: bool = True,
        return_as_string: bool = False,
        include_caption: bool = True,
        include_indices: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        Get a full dialog script for dialogs[index].

        Args:
            index: dialog index in self.dialogs
            include_answer_options: whether to include answer_options
            answer_options_as_text: if True, convert answer option indices to answer text
            return_as_string: if True, return formatted text script instead of dict
            include_caption: whether to include image caption
            include_indices: whether to keep question_idx / answer_idx / gt_index fields

        Returns:
            Dict or formatted string
        """
        self._validate_dialog_index(index)
        item = self.dialogs[index]

        result: Dict[str, Any] = {
            "dialog_index": index,
            "image_id": item.get("image_id"),
            "rounds": []
        }

        if include_caption:
            result["caption"] = item.get("caption")

        rounds = item.get("dialog", [])
        for round_id, turn in enumerate(rounds, start=1):
            question_idx = turn["question"]
            answer_idx = turn["answer"]
            gt_index = turn.get("gt_index")
            answer_option_indices = turn.get("answer_options", [])

            turn_result: Dict[str, Any] = {
                "round_id": round_id,
                "question": self._resolve_question(question_idx),
                "answer": self._resolve_answer(answer_idx),
            }

            if include_indices:
                turn_result["question_idx"] = question_idx
                turn_result["answer_idx"] = answer_idx
                turn_result["gt_index"] = gt_index

            if include_answer_options:
                if answer_options_as_text:
                    turn_result["answer_options"] = self._resolve_answer_options(answer_option_indices)
                else:
                    turn_result["answer_options"] = answer_option_indices

            result["rounds"].append(turn_result)

        if return_as_string:
            return self._format_script(result, include_answer_options=include_answer_options), item.get("image_id")

        return result, item.get("image_id")

    def _format_script(self, script_data: Dict[str, Any], include_answer_options: bool = False) -> str:
        """
        Format structured script data into readable plain text.
        """
        lines: List[str] = []

        lines.append(f"Dialog Index: {script_data['dialog_index']}")
        lines.append(f"Image ID: {script_data['image_id']}")

        if "caption" in script_data:
            lines.append(f"Caption: {script_data['caption']}")

        lines.append("")

        for turn in script_data["rounds"]:
            lines.append(f"[Round {turn['round_id']}]")
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")

            if include_answer_options and "answer_options" in turn:
                lines.append("Answer Options:")
                for i, opt in enumerate(turn["answer_options"]):
                    lines.append(f"  {i}. {opt}")

            lines.append("")

        return "\n".join(lines).strip()

    def get_question_text(self, question_idx: int) -> str:
        return self._resolve_question(question_idx)

    def get_answer_text(self, answer_idx: int) -> str:
        return self._resolve_answer(answer_idx)

    def get_meta(self) -> Dict[str, Any]:
        return {
            "anno_path": str(self.anno_path),
            "split": self.split,
            "version": self.version,
            "num_questions": len(self.questions),
            "num_answers": len(self.answers),
            "num_dialogs": len(self.dialogs),
        }
        
if __name__ == "__main__":
    visdial = VisDialAnnotation("/Users/duypd/MyPC/MyProject/RAGInteractIR/datasets/VisDial/visdial_1.0_train.json")
    
    image_root = "/Users/duypd/MyPC/MyProject/RAGInteractIR/datasets/MSCOCO"
    with open("/Users/duypd/MyPC/MyProject/RAGInteractIR/datasets/VisDial/coco2014_id_to_relpath.json", "r") as f:
        coco_path = json.load(f)
    
    print(visdial.get_meta())
    
    # item = visdial.get_dialog_item(10)
    # print(item)
    
    
    script_text, image_id = visdial.get_script(
        index=375,
        include_answer_options=False,
        return_as_string=True
    )
    
    
    print(f"image path: {coco_path[str(image_id)]}")
    print(script_text)
    
    image = Image.open(os.path.join(image_root, coco_path[str(image_id)]))
    plt.imshow(image)
    plt.show()
    