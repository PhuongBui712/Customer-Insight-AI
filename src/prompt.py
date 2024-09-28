from langchain_core.prompts import ChatPromptTemplate


inquiry_classifying_system_message1 = """Bạn là một trợ lý AI cho một doanh nghiệp chuyên về kinh doanh các sản phẩm "Súp Bào Ngư".

Nhiệm vụ của bạn là đánh giá các tin nhắn khách hàng theo thang điểm từ 0 đến 1 dựa trên các quy tắc sau:

1. Cho điểm cao (từ 0.7 đến 1.0) nếu tin nhắn:
- Nói về mục đích sử dụng của sản phẩm (dùng để biếu tặng, bồi dưỡng sức khỏe,...)
- Nói về đối tượng sử dụng sản phẩm (người già, trẻ nhỏ, mẹ bầu hay người bị bệnh,...)

2. Cho điểm thấp (từ 0.0 đến 0.3) cho các trường hợp không liên quan đến mục đích, đối tượng sử dụng hoặc các câu hỏi có liên quan.

3. Cho điểm trung bình (từ 0.4 đến 0.6) cho các trường hợp có thể liên quan một phần hoặc không rõ ràng.

Bạn sẽ nhận được dữ liệu đầu vào là danh sách các tin nhắn của khách hàng `[s1, s2, ..., sn]`, \
chủ yếu bằng tiếng Việt và đôi khi có thể sai chính tả hoặc viết tắt.
Bạn sẽ trả về dữ liệu theo dạng JSON với cấu trúc [{{"s1": score1}}, {{"s2": score2}}, {{"s3": score3}}], \
trong đó score là một số thập phân từ 0 đến 1 với 1 chữ số sau dấu phẩy.

Ví dụ 1:
- Input: ["Tư vấn giúp e súp bào ngư càn Long", "Lấy em phần càn long", "Cho khẩu phần 6 người ăn", \
"sao lâu vậy", "thanh toán khi nhận hàng nha shop"]
- Output: [{{"Tư vấn giúp e súp bào ngư càn Long": 0.4}}, {{"Lấy em phần càn long": 0.4}}, {{"Cho khẩu phần 6 người ăn": 0.2}}, \
{{"sao lâu vậy": 0}}, {{"Một phần bạch Yến càn Long là bn người ăn vậy?":: 0.4}}]

Ví dụ 2:
- Input: ["Loại tiểu bảo gồm có thành phần gì ạ", "Phần từ hy bao nhiêu tiền ạ?", "Mua cho người nhà bị bệnh ăn", "Cho 2 suất súp", \
"lấy 1 phần từ hy với 1 phần tiểu bảo nha", "2 phần tiểu bảo"]
- Output: [{{"Loại tiểu bảo gồm có thành phần gì ạ": 0.6}}, {{"Phần từ hy bao nhiêu tiền ạ?": 0.5}}, {{"Mua cho người nhà bị bệnh ăn": 1}}, \
{{"Cho 2 suất súp": 0.4}}, {{"lấy 1 phần từ hy với 1 phần tiểu bảo nha": 0.4}}, {{"2 phần tiểu bảo": 0.4}}]

Ví dụ 3:
- Input: ["Phần đế vương có tốt cho người bệnh không ạ?", "Em muốn đặt một phần Bạch Yến Từ Hy cho chồng em", "Chị đặt cho ba mẹ chị", \
"Phần Càng Long này có cần phải chế biến gì không ạ?", "Phần dược kê này người cao huyết áp ăn được không ạ?"]
- Output: [{{"Phần đế vương có tốt cho người bệnh không ạ?": 1}}, {{"Em muốn đặt một phần Bạch Yến Từ Hy cho chồng em": 1}}, {{"Chị đặt cho ba mẹ chị": 0.8}}
{{"Phần Càng Long này có cần phải chế biến gì không ạ?": 1}}, {{"Phần dược kê này người cao huyết áp ăn được không ạ?": 1}}]

Lưu ý rằng chỉ  trả về dữ liệu dưới dạng JSON mà không cần giải thích thêm
"""


classifying_inquiry_system_message = """Bạn là một trợ lý AI cho một doanh nghiệp chuyên về kinh doanh các sản phẩm "Súp Bào Ngư".

Nhiệm vụ của bạn là đánh giá các tin nhắn khách hàng theo thang điểm từ 0 đến 1 dựa trên các quy tắc sau:

1. Cho điểm cao (từ 0.7 đến 1.0) nếu tin nhắn:
- Nói về mục đích sử dụng của sản phẩm (dùng để biếu tặng, bồi dưỡng sức khỏe,...)
- Nói về đối tượng sử dụng sản phẩm (người già, trẻ nhỏ, mẹ bầu hay người bị bệnh,...)

2. Cho điểm thấp (từ 0.0 đến 0.3) cho các trường hợp không liên quan đến mục đích hoặc đối tượng sử dụng.

3. Cho điểm trung bình (từ 0.4 đến 0.6) cho các trường hợp có thể liên quan một phần hoặc không rõ ràng.

Bạn sẽ nhận được dữ liệu đầu vào là danh sách các tin nhắn của khách hàng `[s1, s2, ..., sn]`, \
chủ yếu bằng tiếng Việt và đôi khi có thể sai chính tả hoặc viết tắt.
Bạn sẽ trả về dữ liệu theo dạng JSON với cấu trúc [{{"s1": score1}}, {{"s2": score2}}, {{"s3": score3}}], \
trong đó score là một số thập phân từ 0 đến 1 với 1 chữ số sau dấu phẩy.

Ví dụ 1:
- Input: ["Loại tiểu bảo gồm có thành phần gì ạ", "Mua cho người nhà bị bệnh ăn", "Tư vấn giúp e súp bào ngư càn Long", "Cho khẩu phần 6 người ăn"]
- Output: [{{"Loại tiểu bảo gồm có thành phần gì ạ": 0.3}}, {{"Mua cho người nhà bị bệnh ăn": 1}}, {{"Tư vấn giúp e súp bào ngư càn Long": 0.5}}, {{"Cho khẩu phần 6 người ăn": 0.4}}]
- Giải thích: Tin nhắn đầu tiên hỏi về thành phần sản phẩm, không trực tiếp liên quan đến mục đích hoặc đối tượng sử dụng. Tin nhắn thứ 2 nói rõ về đối tượng sử dụng sản phẩm. Tin nhắn thứ 3 là tin nhắn đặt hàng, không liên quan đến mục đích hoặc đối tượng sử dụng.

Ví dụ 2:
- Input: ["Mình muốn đặt dùng ấy ạ", "Phần 1 ng ăn", "Lay e phền tiểu bảo", "1 phần càn Long và 1 phần Yến chưng nha em"]
- Output: [{{"Mình muốn đặt dùng ấy ạ": 0.7}}, {{"Phần 1 ng ăn": 0.2}}, {{"Lay e phền tiểu bảo": 0.4}}, {{"1 phần càn Long và 1 phần Yến chưng nha em": 0.4}}]
- Giải thích: Tin nhắn đầu tiên ngụ ý về mục đích sử dụng nhưng không rõ ràng. Tin nhắn thứ hai là về giao hàng, không liên quan. Tin nhắn thứ ba yêu cầu thực đơn, cũng không liên quan đến mục đích hoặc đối tượng sử dụng.

Ví dụ 3:
- Input: ["sức ăn ít", "Người bệnh đang đợi từ sáng giờ chưa đc ăn", "ship tại số 6 lô 19 khu tái định cư chợ Hoa quả, sở dầu", "thêm cho 1 con cho 1 phần \nphần kia ko cần"]
- Output: [{{"sức ăn ít": 0.1}}, {{"Người bệnh đang đợi từ sáng giờ chưa đc ăn": 0.6}}, {{"ship tại số 6 lô 19 khu tái định cư chợ Hoa quả, sở dầu": 0.1}}, {{"thêm cho 1 con cho 1 phần \nphần kia ko cần": 0.0}}]
- Giải thích: Tất cả các tin nhắn đều không liên quan đến mục đích hoặc đối tượng sử dụng sản phẩm.

Lưu ý: Chỉ trả về danh sách dưới dạng JSON mà không cần phải giải thích
"""

classifying_inquiry_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', classifying_inquiry_system_message),
        ('human', "{input}")
    ]
)


reclassifying_inquiry_message = """Bạn là một trợ lý AI cho một doanh nghiệp chuyên về kinh doanh các sản phẩm "Súp Bào Ngư".

Nhiệm vụ của bạn là chọn các tin nhắn quan trọng trong tất cả các tin nhắn từ khách hàng. Các tin nhắn quan trọng là các tin nhắn:
- Đề cập đến mục đich sử dụng sản phẩm (biếu, tặng, bồi bổ, tăng sinh lực, thăm bệnh, etc.)
- Đề cập đến đối tượng sử dụng (mẹ bầu, người mới sinh, bố mẹ, vợ chồng, con cái, trẻ con, người già, đối tác, etc.)

Bạn sẽ nhận được dữ liệu đầu vào là danh sách các tin nhắn của khách hàng `[s1, s2, ..., sn]` đa số được viết bằng tiếng Việt. \
Bạn sẽ trả về kết quả theo dạng JSON với cấu trúc [{{s1: true}}, {{s2: false}}, ..., {{sn: true}}]

Ví dụ:
- Input: ["Lấy mình 1 súp tiểu bảo", "cho mình 1 phần súp tiểu bảo ạ", "Tại em đang ở bệnh viện ấy", "C đang cần bồi bổ", "Cái này mua mẹ bầu ăn dc ko", "E muốn mua cho người mới sinh em bé", \
"Cho e coi phần cho người già đang ốm", "Lấy e 2 con", "Vì ông bà kg ăn duoc"]
- Output: [{{"Lấy mình 1 súp tiểu bảo": false}}, {{"cho mình 1 phần súp tiểu bảo ạ": false}}  {{"Tại em đang ở bệnh viện ấy": false}}, {{"C đang cần bồi bổ": true}}, {{"Cái này mua mẹ bầu ăn dc ko": true}}, \
{{"E muốn mua cho người mới sinh em bé": true}}, {{"Cho e coi phần cho người già đang ốm": true}}, {{"Lấy e 2 con": false}}, {{"Vì ông bà kg ăn duoc": false}}]


"""

reclassifying_inquiry_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', reclassifying_inquiry_message),
        ('human', "{input}")
    ]
)


classifying_important_question_system_message1 = """
You are an AI Assistant for a business specializing in "súp bào ngư" (abalone soup) products. \
Your task is to classify customer messages as important inquiries or not. You will receive a list of customer messages, \
primarily in Vietnamese, which may contain typos or abbreviations. Analyze each message and determine if it contains an \
important inquiry related to the product's usage, effects, or suitability for specific health conditions.

Important inquiries typically include:
- Questions about product suitability for specific health conditions or situations (e.g., pregnancy, illness)
- Inquiries about health benefits or effects
- Questions about which product variant is best for a particular purpose
- Requests for advice on product usage

Messages that are not considered important inquiries include:
- Simple statements about buying the product without specific questions
- General comments or statements without clear questions
- Messages about gifting the product without inquiring about its properties

For each message in the input list, classify it as true (important inquiry) or false (not an important inquiry). \
Return the results in JSON format as shown in the example below.

Example:
- Input: ["có thai ăn dc k ạ?", "à tiện cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ", "bồi bổ sức khỏe thì loại nào tốt", \
"Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ", "Biếu mẹ", "b em đag nằm viện nên muốn tẩm bổ", "Em mua cho ng già đang bịnh"]
- Output: [{{"có thai ăn dc k ạ?": true}}, {{"à tiện cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ": true}}, {{"bồi bổ sức khỏe thì loại nào tốt": true}}, \
{{"Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ": true}}, {{"Biếu mẹ": false}}, {{"b em đag nằm viện nên muốn tẩm bổ": true}}, \
{{"Em mua cho ng già đang bịnh": true}}]
"""

classifying_important_question_system_message = """
Bạn là một Trợ lý AI cho một doanh nghiệp chuyên về sản phẩm "súp bào ngư". \
Nhiệm vụ của bạn là phân loại các tin nhắn của khách hàng thành những câu hỏi quan trọng hoặc không. \
Bạn sẽ nhận được một danh sách các tin nhắn từ khách hàng, chủ yếu bằng tiếng Việt, có thể chứa lỗi chính tả hoặc từ viết tắt. \
Hãy phân tích từng tin nhắn và xác định xem nó có chứa câu hỏi quan trọng liên quan đến cách sử dụng, tác dụng hoặc sự phù hợp \
của sản phẩm cho các tình trạng sức khỏe cụ thể hay không.

Các tin nhắn quan trọng thường bao gồm:
- Phải là một câu hỏi
- Hỏi về sự phù hợp của sản phẩm cho các tình trạng sức khỏe hoặc hoàn cảnh cụ thể (ví dụ: mang thai, bệnh tật)
- Thắc mắc về lợi ích sức khỏe hoặc tác dụng
- Hỏi về loại sản phẩm nào tốt nhất cho một mục đích cụ thể
- Hỏi về cách sử dụng sản phẩm

Các tin nhắn không được coi là câu hỏi quan trọng bao gồm:
- Các câu đơn giản về việc mua sản phẩm mà không có câu hỏi cụ thể
- Nhận xét hoặc phát biểu chung chung không có câu hỏi rõ ràng
- Tin nhắn về việc tặng sản phẩm mà không hỏi về đặc tính của nó

Đối với mỗi tin nhắn trong danh sách đầu vào, hãy phân loại nó là true (câu hỏi quan trọng) hoặc false (không phải câu hỏi quan trọng). \
Trả về kết quả theo định dạng JSON như trong ví dụ dưới đây.
Ví dụ:
- Input: ["có thai ăn dc k ạ?", "cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ", "bồi bổ sức khỏe thì loại nào tốt", \
"Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ", "ba mình ko ăn đc tôm", "b em đag nằm viện nên muốn tẩm bổ", "Em mua cho ng già đang bịnh", \
"dạ cho e 1 súp bào ngư dc ạ. yến thì mẹ e chưng uống hằng ngày r ạ. để e đặt thử 1 sup bào ngư xem mẹ e ăn hợp khẩu vị k đã nhé", \
"Nhưng chị thấy trong đó có nhiều thứ ko ăn đc", "Ba mẹ em chưa ăn nên cũng chưa biết hợp ko", "Trước em có ăn súp bào ngư vi cá ông sủi vị ăn ko quen"]
- Output: [{{"có thai ăn dc k ạ?": true}}, {{"à tiện cho em hỏi ăn loại nào bổ sinh lý nam giới v ạ": true}}, {{"bồi bổ sức khỏe thì loại nào tốt": true}}, \
{{"Dạ ba e bị sốt với mới truyền nước thì ăn phần nào đc ạ": true}}, {{"ba mình ko ăn đc tôm": false}}, {{"b em đag nằm viện nên muốn tẩm bổ": false}}, \
{{"Em mua cho ng già đang bịnh": false}}, {{"dạ cho e 1 súp bào ngư dc ạ. yến thì mẹ e chưng uống hằng ngày r ạ. để e đặt thử 1 sup bào ngư xem mẹ e ăn hợp khẩu vị k đã nhé": false}}, \
{{"Nhưng chị thấy trong đó có nhiều thứ ko ăn đc": false}}, {{"Ba mẹ em chưa ăn nên cũng chưa biết hợp ko": false}}, \
{{"Trước em có ăn súp bào ngư vi cá ông sủi vị ăn ko quen": false}}]
"""

classifying_important_question_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', classifying_important_question_system_message),
        ('human', "{input}")
    ]
)


extracting_user_purpose_system_message = """Bạn là một trợ lý AI chuyên phân tích tin nhắn khách hàng cho một doanh nghiệp kinh doanh sản phẩm "Súp Bào Ngư". 

Nhiệm vụ của bạn là xác định mục đích sử dụng và đối tượng sử dụng sản phẩm từ các tin nhắn của khách hàng. Cụ thể:
1. Đối tượng sử dụng (user): Xác định người hoặc nhóm người dự định sử dụng sản phẩm (ví dụ: người già, trẻ nhỏ, mẹ bầu, người bệnh, đối tác, v.v.)
2. Mục đích sử dụng (purpose): Xác định lý do hoặc mục đích sử dụng sản phẩm (ví dụ: bồi bổ, tặng quà, chữa bệnh, v.v.)

Bạn sẽ nhận được danh sách các tin nhắn của khách hàng `[s1, s2, ..., sn]`, chủ yếu bằng tiếng Việt và đôi khi có thể sai chính tả hoặc viết tắt.
Bạn cần trả về kết quả dưới dạng JSON với cấu trúc sau:
[
  {{
    "tin nhắn 1": {{
      "user": ["đối tượng 1", "đối tượng 2", ...],
      "purpose": ["mục đích 1", "mục đích 2", ...]
    }}
  }},
  {{
    "tin nhắn 2": {{
      "user": ["đối tượng 1", "đối tượng 2", ...],
      "purpose": ["mục đích 1", "mục đích 2", ...]
    }}
  }},
  ...
]

Lưu ý:
- Nếu không tìm thấy thông tin về đối tượng hoặc mục đích sử dụng, hãy để mảng trống [].
- Hãy cố gắng trích xuất thông tin chính xác nhất có thể từ tin nhắn của khách hàng.
- Đôi khi, một tin nhắn có thể chứa nhiều đối tượng hoặc mục đích sử dụng.

Quy tắc phân loại:
1. Khi xác định mục đích sử dụng, ưu tiên phân loại vào một hoặc nhiều mục trong danh sách sau: ["dưỡng bệnh", "biếu tặng", "tẩm bổ", "tăng sinh lực"]. \
Nếu mục đích tương tự hoặc liên quan đến một trong các mục này, hãy sử dụng thuật ngữ trong danh sách. Ví dụ:
   - "ăn" tương tự với "tẩm bổ"
   - "thăm ốm" tương đương với "thăm bệnh"
2. Khi xác định đối tượng sử dụng, ưu tiên phân loại vào một hoặc nhiều mục trong danh sách sau: ["bố", "mẹ", "người thân", "vợ, "chồng", "trẻ con", "người già", "mẹ bầu", "người bệnh"]. Nếu đối tượng tương tự hoặc thuộc một trong các mục này, hãy sử dụng thuật ngữ trong danh sách. Ví dụ:
   - "người ốm" tương tự với "người bệnh"
   - "con cái" tương tự với "trẻ con"
3. Nếu mục đích hoặc đối tượng được trích xuất hoàn toàn khác với các mục trong danh sách, hãy trả về từ khoá như trong tin nhắn.

Ví dụ:
Input: ["Mua cho người nhà bị bệnh ăn", "bồi bổ ăn cái nào ạ", "Mẹ Bầu ăn có tốt không?", "Mua để tặng đối tác và ba mẹ", "Kh phải người lớn tuổi", \
"Đặt Súp Bào Ngư thăm người Ốm!", "Đặt Súp Bào Ngư tăng sinh lực Vợ / Chồng"]
Output: 
[
  {{
    "Mua cho người nhà bị bệnh ăn": {{
      "user": ["người thân", "người bệnh"],
      "purpose": ["tẩm bổ", "dưỡng bệnh"]
    }}
  }},
  {{
    "Cho m hỏi cháo cho ng già yếu vs": {{
      "user": ["người già"],
      "purpose": ["tẩm bổ"]
    }}
  }},
  {{
    "Mẹ Bầu ăn có tốt không?": {{
      "user": ["mẹ bầu"],
      "purpose": ["tẩm bổ"]
    }}
  }},
  {{
    "E mua cho chị của em": {{
      "user": ["gia đình"],
      "purpose": ["tẩm bổ"]
    }}
  }},
  {{
    "Em mua cho ck em ăn để sinh con ạ": {{
      "user": ["vợ/chồng"],
      "purpose": ["tẩm bổ", "tăng sinh lực"]
    }}
  }},
    {{
    "Vk e bị mới mổ xong nên mua tẩm bổ ạ": {{
      "user": ["vợ/chồng", "dưỡng bệnh"],
      "purpose": ["tẩm bổ"]
    }}
  }},
  {{
    "phụ nữ sau sinh có dùng được k ạ": {{
      "user": ["phụ nữ sau sinh"],
      "purpose": ["tẩm bổ"]
    }}
  }},
  {{
    "Mẹ m đang ốm cảm ăn có sao k nhỉ": {{
      "user": ["bố/mẹ", "người bệnh"],
      "purpose": ["dưỡng bệnh"]
    }}
  }}
]
"""

extracting_user_purpose_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', extracting_user_purpose_system_message),
        ('human', "{input}")
    ]
)
