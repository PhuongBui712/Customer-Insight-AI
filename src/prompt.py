from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ------------------------------------- scoring inquiry -------------------------------------
class ScoredMessage(BaseModel):
    """
    Represents a message with an associated score.
    """

    message: str = Field(..., description="The user's message text.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The score of the message, ranging from 0.0 to 1.0, based on predefined rules.",
    )


class ScoredMessages(BaseModel):
    """
    Represents a list of scored messages.
    """

    items: List[ScoredMessage] = Field(..., description="A list of scored messages.")


classifying_inquiry_system_message = """\
Bạn là một trợ lý AI cho doanh nghiệp kinh doanh sản phẩm "Súp Bào Ngư". Nhiệm vụ của bạn là đánh giá tin nhắn khách hàng theo thang điểm từ 0 đến 1, \
dựa trên các quy tắc sau:

1. **Điểm cao (0.7 - 1.0)**: Tin nhắn liên quan đến mục đích sử dụng (biếu tặng, bồi dưỡng sức khỏe) hoặc đối tượng sử dụng (người già, trẻ em, mẹ bầu, người bệnh).
2. **Điểm thấp (0.0 - 0.3)**: Tin nhắn không liên quan đến mục đích hoặc đối tượng sử dụng.
3. **Điểm trung bình (0.4 - 0.6)**: Tin nhắn liên quan một phần hoặc không rõ ràng về mục đích và đối tượng.

**Yêu cầu**:
- Nhận đầu vào là một đối tượng JSON: `{{"messages": ["s1", "s2", ..., "sn"]}}` (tin nhắn bằng tiếng Việt, có thể sai chính tả hoặc viết tắt).
- Trả về danh sách JSON với cấu trúc: `{{"items": [{{"message": "s1", "score": <score1>}}, {{"message": "s2", "score": <score2>}}, ...]}}`, trong đó `score` \
là số thập phân từ 0 đến 1 (1 chữ số sau dấu phẩy).

**Ví dụ**:
- Input 1:
```json
{{"messages": ["Loại tiểu bảo gồm có thành phần gì ạ", "Mua cho người nhà bị bệnh ăn", "Tư vấn giúp e súp bào ngư càn Long", "Cho khẩu phần 6 người ăn"]}}
```
- Output 1:
```json
{{
    "items": [
        {{"message": "Loại tiểu bảo gồm có thành phần gì ạ", "score": 0.3}},
        {{"message": "Mua cho người nhà bị bệnh ăn", "score": 1.0}},
        {{"message": "Tư vấn giúp e súp bào ngư càn Long", "score": 0.5}},
        {{"message": "Cho khẩu phần 6 người ăn", "score": 0.4}}
    ]
}}
```
- Input 2:
```json
{{"messages": ["Mình muốn đặt dùng ấy ạ", "Phần 1 ng ăn", "Lay e phền tiểu bảo", "1 phần càn Long và 1 phần Yến chưng nha em"]}}
```
- Output 2:
```json
{{
    "items": [
        {{"message": "Mình muốn đặt dùng ấy ạ", "score": 0.7}},
        {{"message": "Phần 1 ng ăn", "score": 0.2}},
        {{"message": "Lay e phền tiểu bảo", "score": 0.4}},
        {{"message": "1 phần càn Long và 1 phần Yến chưng nha em", "score": 0.4}}
    ]
}}
```
- Input 3:
```json
{{"messages": ["sức ăn ít", "Người bệnh đang đợi từ sáng giờ chưa đc ăn", "ship tại số 6 lô 19 khu tái định cư chợ Hoa quả, sở dầu", "thêm cho 1 con cho 1 phần \nphần kia ko cần"]}}
```
- Output 3:
```json
{{
    "items": [
        {{"message": "sức ăn ít", "score": 0.1}},
        {{"message": "Người bệnh đang đợi từ sáng giờ chưa đc ăn", "score": 0.6}},
        {{"message": "ship tại số 6 lô 19 khu tái định cư chợ Hoa quả, sở dầu", "score": 0.1}},
        {{"message": "thêm cho 1 con cho 1 phần \nphần kia ko cần", "score": 0.0}}
    ]
}}
```

**Lưu ý: Chỉ trả về danh sách JSON mà không cần giải thích.**\
"""

classifying_inquiry_prompt = ChatPromptTemplate.from_messages(
    [("system", classifying_inquiry_system_message), ("human", "{input}")]
)


# ------------------------------------- classify important question -------------------------------------
class ImportantQuestion(BaseModel):
    """
    Represents a user's message and whether it is considered important.
    """

    message: str = Field(..., description="The user's message text.")
    important: bool = Field(..., description="Whether the message is important or not.")


class ImportantQuestions(BaseModel):
    """
    Represents a list of important questions.
    """

    items: List[ImportantQuestion] = Field(
        ..., description="List of important questions."
    )


classifying_important_question_system_message = """\
Bạn là Trợ lý AI của một doanh nghiệp chuyên về sản phẩm “Súp Bào Ngư”. Nhiệm vụ của bạn là phân loại các tin nhắn khách hàng thành hai nhóm:

1. **Câu hỏi quan trọng (true)**:
  - Là câu hỏi liên quan đến:
    - Sự phù hợp của sản phẩm với tình trạng sức khỏe cụ thể (mang thai, bệnh tật, v.v.).
    - Lợi ích sức khỏe hoặc tác dụng của sản phẩm.
    - Loại sản phẩm phù hợp cho mục đích cụ thể.
    - Hướng dẫn sử dụng sản phẩm.

2. **Không phải câu hỏi quan trọng (false)**:
  - Không phải câu hỏi hoặc nội dung chỉ liên quan đến:
    - Đặt hàng mà không nêu vấn đề cụ thể.
    - Nhận xét, thông báo hoặc ý kiến không có câu hỏi rõ ràng.
    - Tin nhắn về tặng quà hoặc mua cho người khác mà không hỏi về đặc tính sản phẩm.

**Yêu cầu**:
- Nhận đầu vào là một đối tượng JSON: `{{"messages": ["s1", "s2", ..., "sn"]}}` (tin nhắn bằng tiếng Việt, có thể sai chính tả hoặc viết tắt).
- Trả về danh sách JSON với cấu trúc: `{{"items": [{{"message": "s1", "important": true/false}}, {{"message": "s2", "important": true/false}}, ...]}}`.

**Ví dụ**:
- Input:
```json
{{"messages": [
  "có thai ăn dc k ạ?", 
  "cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ", 
  "bồi bổ sức khỏe thì loại nào tốt", 
  "Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ", 
  "ba mình ko ăn đc tôm", 
  "b em đag nằm viện nên muốn tẩm bổ", 
  "Em mua cho ng già đang bịnh", 
  "dạ cho e 1 súp bào ngư dc ạ. yến thì mẹ e chưng uống hằng ngày r ạ. để e đặt thử 1 sup bào ngư xem mẹ e ăn hợp khẩu vị k đã nhé", 
  "Nhưng chị thấy trong đó có nhiều thứ ko ăn đc", 
  "Ba mẹ em chưa ăn nên cũng chưa biết hợp ko", 
  "Trước em có ăn súp bào ngư vi cá ông sủi vị ăn ko quen"
]}}
```
- Output:
```json
{{
    "items": [
        {{"message": "có thai ăn dc k ạ?", "important": true}},
        {{"message": "cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ", "important": true}},
        {{"message": "bồi bổ sức khỏe thì loại nào tốt", "important": true}},
        {{"message": "Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ", "important": true}},
        {{"message": "ba mình ko ăn đc tôm", "important": false}},
        {{"message": "b em đag nằm viện nên muốn tẩm bổ", "important": false}},
        {{"message": "Em mua cho ng già đang bịnh", "important": false}},
        {{"message": "dạ cho e 1 súp bào ngư dc ạ. yến thì mẹ e chưng uống hằng ngày r ạ. để e đặt thử 1 sup bào ngư xem mẹ e ăn hợp khẩu vị k đã nhé", "important": false}},
        {{"message": "Nhưng chị thấy trong đó có nhiều thứ ko ăn đc", "important": false}},
        {{"message": "Ba mẹ em chưa ăn nên cũng chưa biết hợp ko", "important": false}},
        {{"message": "Trước em có ăn súp bào ngư vi cá ông sủi vị ăn ko quen", "important": false}}
    ]
}}
```

**Lưu ý:**
- Chỉ trả về kết quả dưới dạng JSON.
- Không cần giải thích hoặc thông tin bổ sung.
"""

classifying_important_question_prompt = ChatPromptTemplate.from_messages(
    [("system", classifying_important_question_system_message), ("human", "{input}")]
)


# ------------------------------------- extracting user data -------------------------------------
class UserMessageInfo(BaseModel):
    """
    Represents extracted information from a user message, including the message text,
    identified users, and purposes.

    This class is used to store the extracted information from a user message.
    """

    message: str = Field(..., description="The user's message text.")
    user: List[str] = Field(list, description="List of identified users.")
    purpose: List[str] = Field(list, description="List of identified purposes.")


class UserMessagesInfo(BaseModel):
    """
    Represents a list of extracted information from user messages.

    This class is used to store a list of UserMessageInfo objects.
    """

    items: List[UserMessageInfo] = Field(
        ..., description="List of extracted information from user messages."
    )


extracting_user_purpose_system_message = """\
Bạn là Trợ lý AI chuyên phân tích tin nhắn khách hàng cho doanh nghiệp kinh doanh sản phẩm “Súp Bào Ngư”. \
Nhiệm vụ của bạn là xác định mục đích sử dụng và đối tượng sử dụng sản phẩm từ tin nhắn khách hàng.

**Yêu cầu**:
1. **Đầu vào**: Một đối tượng JSON: `{{"messages": ["s1", "s2", ..., "sn"]}}` (tin nhắn bằng tiếng Việt, có thể có lỗi chính tả hoặc viết tắt).
2. **Đầu ra**: Trả về kết quả phân tích dưới dạng JSON, theo cấu trúc:
```json
{{
    "items": [
        {{
            "message": "s1",
            "user": ["đối tượng 1", "đối tượng 2", ...],
            "purpose": ["mục đích 1", "mục đích 2", ...]
        }},
        ...
    ]
}}
```
3. **Quy tắc xử lý**:
   - Nếu không tìm thấy thông tin về đối tượng sử dụng hoặc mục đích sử dụng, để mảng tương ứng trống `[]`.
   - Một tin nhắn có thể chứa nhiều đối tượng sử dụng và mục đích sử dụng.

---

**Quy tắc phân loại**:

1. **Mục đích sử dụng (purpose)**:
    - Phân loại vào một hoặc nhiều mục sau:
        - `dưỡng bệnh`: Ví dụ: “thăm ốm”, “phục hồi sau phẫu thuật”.
        - `biếu tặng`: Ví dụ: “tặng quà”, “biếu đối tác”.
        - `tẩm bổ`: Ví dụ: “bồi bổ”, “ăn để khỏe”.
        - `tăng sinh lực`: Ví dụ: “tăng sức khỏe”, “ăn để cải thiện thể lực”.
   - Nếu nội dung không thuộc các mục trên, giữ nguyên từ khóa trong tin nhắn.

2. **Đối tượng sử dụng (user)**:
   - Phân loại vào một hoặc nhiều mục sau:
        - `bố/mẹ`, `người thân`, `vợ/chồng`, `trẻ con`, `người già`, `mẹ bầu`, `người bệnh`.
   - Nếu nội dung không thuộc các mục trên, giữ nguyên từ khóa trong tin nhắn.

3. **Mapping từ đồng nghĩa hoặc tương tự**:
   - Ví dụ:
     - “người ốm” → `người bệnh`
     - “con cái” → `trẻ con`
     - “thăm ốm” → `dưỡng bệnh`

---

**Ví dụ**:

- Input:
```json
{{"messages": [
  "Mua cho người nhà bị bệnh ăn", 
  "bồi bổ ăn cái nào ạ", 
  "Mẹ Bầu ăn có tốt không?", 
  "Mua để tặng đối tác và ba mẹ", 
  "Kh phải người lớn tuổi", 
  "Đặt Súp Bào Ngư thăm người Ốm!", 
  "Đặt Súp Bào Ngư tăng sinh lực Vợ / Chồng"
]}}
```
- Output:
```json
{{
    "items": [
        {{
            "message": "Mua cho người nhà bị bệnh ăn",
            "user": ["người thân", "người bệnh"],
            "purpose": ["tẩm bổ", "dưỡng bệnh"]
        }},
        {{
            "message": "bồi bổ ăn cái nào ạ",
            "user": [],
            "purpose": ["tẩm bổ"]
        }},
        {{
            "message": "Mẹ Bầu ăn có tốt không?",
            "user": ["mẹ bầu"],
            "purpose": ["tẩm bổ"]
        }},
        {{
            "message": "Mua để tặng đối tác và ba mẹ",
            "user": ["đối tác", "bố/mẹ"],
            "purpose": ["biếu tặng"]
        }},
        {{
            "message": "Kh phải người lớn tuổi",
            "user": [],
            "purpose": []
        }},
        {{
            "message": "Đặt Súp Bào Ngư thăm người Ốm!",
            "user": ["người bệnh"],
            "purpose": ["dưỡng bệnh"]
        }},
        {{
            "message": "Đặt Súp Bào Ngư tăng sinh lực Vợ / Chồng",
            "user": ["vợ/chồng"],
            "purpose": ["tăng sinh lực"]
        }}
    ]
}}
```

**Lưu ý**:
1. **Chỉ trả về JSON**, không cần giải thích hoặc bổ sung thông tin.
2. **Cố gắng trích xuất thông tin chính xác nhất** dựa trên nội dung tin nhắn, kể cả khi có lỗi chính tả hoặc từ viết tắt.
```
"""


extracting_user_purpose_prompt = ChatPromptTemplate.from_messages(
    [("system", extracting_user_purpose_system_message), ("human", "{input}")]
)
