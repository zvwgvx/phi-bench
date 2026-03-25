# PhiBench Internal Research Spec

Status: internal design document

Last updated: 2026-03-24

Owner: project note rewritten from the original 30-minute draft into an implementable research specification

## 1. Mục tiêu tài liệu

Tài liệu này không phải paper public. Đây là bản đặc tả nội bộ để biến ý tưởng `PhiBench` thành một hệ thống thí nghiệm có thể triển khai, kiểm tra, và bác bỏ nếu cần.

Mục tiêu của tài liệu:

1. Giữ lại trực giác lõi của ý tưởng ban đầu:
   nếu mô hình thật sự dùng cấu trúc logic, tín hiệu nội tại của nó phải tương đối ổn định khi bề mặt bài toán thay đổi nhưng cấu trúc giữ nguyên.
2. Siết lại các định nghĩa để không overclaim.
3. Thay các chỗ mơ hồ bằng metric, quy trình, và giả thuyết có thể kiểm định.
4. Tạo nền tảng cho code implementation trên `Llama-3.2-1B`.

## 2. Câu hỏi nghiên cứu

PhiBench muốn trả lời câu hỏi sau:

> Có tồn tại một chỉ số nội tại đo từ hidden states của Transformer, đủ ổn định và đủ dự báo, để phân biệt giữa hai chế độ xử lý:
> 
> 1. xử lý dựa trên cấu trúc logic tổng quát hóa được
> 2. xử lý dựa trên nhận diện mẫu bề mặt

PhiBench không cố chứng minh:

1. mô hình "có ý thức"
2. mô hình "suy luận thật" theo nghĩa triết học mạnh
3. integrated information theo đúng IIT full formalism đã được tính chính xác

PhiBench chỉ cố kiểm tra giả thuyết thực dụng hơn:

> Một integration proxy nội tại, nếu được định nghĩa tốt, có thể dự báo hành vi tổng quát hóa tốt hơn các benchmark chỉ nhìn đầu ra.

## 3. Điều chỉnh framing so với bản note gốc

Ý tưởng gốc gắn trực tiếp với `Phi` của Integrated Information Theory. Đây là trực giác tốt để khởi phát dự án, nhưng nếu giữ nguyên claim đó thì specification bị hở ở ba điểm:

1. `Phi` chuẩn trong IIT là đại lượng trên một hệ động lực có state transition và minimum information partition rõ ràng.
2. Hidden states của Transformer trong forward pass không tự động thỏa các giả định đó.
3. Metric covariance tĩnh trên một tensor tại cùng một lớp không phải là exact IIT `Phi`.

Vì vậy, trong tài liệu này:

1. Ta dùng tên `Phi-proxy` cho metric chính.
2. Ta chỉ dùng từ `IIT-inspired` hoặc `lấy cảm hứng từ IIT`.
3. Ta chỉ quay lại từ `Phi` khi nói rõ đây là `Phi-proxy`, không phải exact IIT Phi.

## 4. Luận điểm lõi của PhiBench

Luận điểm lõi của dự án là:

1. Nếu mô hình giải bài bằng nhận diện bề mặt, tín hiệu nội tại sẽ phụ thuộc mạnh vào wording, ngữ cảnh, và lexical priors.
2. Nếu mô hình giải bài bằng cấu trúc, profile xử lý nội tại sẽ ổn định hơn qua các biến thể cùng structure.
3. Độ chính xác đầu ra một mình không phân biệt được hai trường hợp này.

Vì vậy, benchmark phải đo đồng thời:

1. `Accuracy`
2. `Generalization under symmetric transformations`
3. `Internal integration profile`
4. `Stability của profile đó qua các biến thể`

## 5. Giả thuyết nghiên cứu

### H1. Structure invariance hypothesis

Với các bài toán cùng cấu trúc logic nhưng khác bề mặt, profile `Phi-proxy` theo lớp sẽ gần nhau hơn so với các bài toán khác cấu trúc.

### H2. Generalization hypothesis

Trên các family bài toán ngoài phân phối, `PhiStability` cao sẽ tương quan dương với khả năng trả lời đúng.

### H3. Depth sensitivity hypothesis

Với bài toán đòi hỏi nhiều bước hơn, `Phi-proxy` ở các lớp sâu sẽ tăng hoặc dịch đỉnh sang sâu hơn.

### H4. Anti-shortcut hypothesis

Những trường hợp `Accuracy` cao nhưng `PhiStability` thấp là dấu hiệu của shortcut hoặc surface matching, không phải bằng chứng mạnh cho reasoning.

## 6. Phạm vi tuyên bố

Nếu các giả thuyết trên được xác nhận, ta chỉ được phép tuyên bố:

1. `Phi-proxy` là một signal nội tại hữu ích để phân biệt các chế độ xử lý thông tin.
2. Signal này có giá trị dự báo tổng quát hóa ngoài phân phối tốt hơn hoặc bổ sung cho accuracy.
3. PhiBench có thể phát hiện trường hợp "đúng vì lý do sai".

Ta không được tuyên bố:

1. mô hình có exact integrated information theo IIT đã được chứng minh
2. mô hình có consciousness
3. mô hình suy luận theo nghĩa hình thức hoàn chỉnh chỉ dựa vào một scalar metric

## 7. Đối tượng mô hình

Mô hình mặc định:

1. `meta-llama/Llama-3.2-1B`

Lý do chọn:

1. đủ nhỏ để hook toàn bộ layer trên phần cứng tiêu dùng
2. có tokenizer và weight sẵn trên Hugging Face
3. phù hợp làm baseline implementation trước khi mở rộng sang model lớn hơn

## 8. Đơn vị phân tích

Một lỗi lớn trong nhiều thiết kế probing là trộn lẫn các cấp phân tích. PhiBench phải tách rõ:

1. `Task family`
   một cấu trúc logic gốc, ví dụ "tỉ lệ", "phép đổi đơn vị", "so sánh nhanh hơn/chậm hơn"
2. `Task instance`
   một bài cụ thể trong family, với số liệu và wording cụ thể
3. `Symmetric triplet`
   bộ ba gồm:
   `T_original`, `T_surface`, `T_inverse`
4. `Layer state`
   hidden state của token hoặc tập token tại layer `l`
5. `Model decision`
   output cuối cùng của mô hình

Đơn vị để báo cáo thống kê không phải token riêng lẻ. Đơn vị chính là `task instance` hoặc `triplet`.

## 9. Thiết kế benchmark

### 9.1. Nguyên lý chung

Mỗi family phải thỏa:

1. có cấu trúc logic mô tả được bằng template
2. có thể sinh ít nhất ba biến thể giữ nguyên cấu trúc nhưng đổi bề mặt
3. có thể kiểm tra đáp án đúng bằng rule-based verifier
4. có thể sinh dữ liệu đủ lớn để thống kê, không phụ thuộc annotation thủ công nặng

### 9.2. Ba biến thể đối xứng cốt lõi

Mỗi instance sẽ được sinh thành một triplet:

1. `T_original`
   phrasing gần với kiểu bài toán phổ biến trong dữ liệu huấn luyện
2. `T_surface`
   cùng structure nhưng thay bối cảnh, entity, từ vựng, và lexical texture
3. `T_inverse`
   cùng structure cốt lõi nhưng đảo chiều câu hỏi hoặc đổi quan hệ đầu vào/đầu ra

Mục tiêu của triplet:

1. tách cấu trúc logic khỏi bề mặt ngôn ngữ
2. đo xem profile nội tại có bất biến tương đối không

### 9.3. Bốn nhóm bài toán ban đầu

#### Group A. Context transfer

Mục tiêu:

1. giữ structure toán học
2. thay hoàn toàn world context

Ví dụ:

1. táo, hàng hóa, tiền
2. tàu vũ trụ, nhiên liệu, năng lượng
3. vật thể giả tưởng, đơn vị giả tưởng

#### Group B. Chain extension

Mục tiêu:

1. kiểm tra độ nhạy theo số bước suy luận
2. xem `Phi-proxy` có tăng theo depth requirement không

Thiết kế:

1. 2-step
2. 3-step
3. 4-step

#### Group C. Linguistic trap

Mục tiêu:

1. wording gần giống nhau
2. structure logic khác nhau

Ví dụ:

1. "nhanh hơn 20 km/h"
2. "nhanh gấp đôi"
3. "mất ít thời gian hơn"

Nếu profile nội tại gần như giống nhau ở cả ba, mô hình có thể đang trigger lexical shortcut.

#### Group D. OOD symbolic knowledge

Mục tiêu:

1. buộc mô hình dùng định nghĩa vừa cho trong prompt
2. tránh phụ thuộc memory của dữ kiện thế giới

Ví dụ:

1. đơn vị giả tưởng
2. hằng số giả tưởng
3. quy tắc ánh xạ mới được định nghĩa tại chỗ

## 10. Sinh dữ liệu

### 10.1. Cấu trúc dữ liệu chuẩn

Mỗi sample nên có schema gần như sau:

```json
{
  "family_id": "unit_conversion",
  "instance_id": "unit_conversion_000123",
  "split": "test",
  "logic_template": "a * k = b",
  "difficulty": 2,
  "variant": "surface",
  "prompt": "Tren hanh tinh Zyron, 1 zunit = 7 litz. Neu ban co 49 zunit, do la bao nhieu litz?",
  "answer_text": "343",
  "answer_value": 343,
  "metadata": {
    "entities": ["zunit", "litz", "Zyron"],
    "num_steps": 1,
    "surface_style": "fictional_science"
  }
}
```

### 10.2. Ràng buộc dữ liệu

Để giảm leakage và shortcut:

1. tách train, calibration, dev, test theo `logic family` và lexical surface
2. với OOD tasks, entity names nên là synthetic
3. tránh reuse nguyên câu phổ biến trên internet
4. giữ verifier tách biệt khỏi generator

### 10.3. Số lượng khuyến nghị

Cho phase đầu:

1. 4 groups
2. mỗi group 5 families
3. mỗi family 200 instances
4. mỗi instance sinh đủ 3 variants

Tổng:

1. `4 x 5 x 200 x 3 = 12,000 prompts`

Số này đủ để:

1. ước lượng metric theo lớp
2. bootstrap CI
3. so sánh giữa families

## 11. Định nghĩa state trong Transformer

Đây là phần bắt buộc phải rõ để metric có nghĩa.

### 11.1. State cơ sở

Gọi:

1. `h_l(t)` là residual stream vector của token nội dung thứ `t` tại layer `l`

Trong đó:

1. `l` là layer index
2. `t` là token position

### 11.2. Transition được đo

PhiBench dùng transition theo layer, không phải theo thời gian thật:

1. từ `h_l(t)` sang `h_{l+1}(t)`

Điều này có nghĩa:

1. "động lực học" ở đây là quá trình biến đổi thông tin qua depth của mạng
2. ta không tuyên bố đây là temporal dynamics sinh học hay IIT nguyên bản

### 11.3. Token nào được lấy

Không nên dùng toàn bộ token một cách mù quáng. Cần tạo `content mask` để loại:

1. BOS
2. EOS
3. token padding
4. punctuation thuần
5. token của phần instruction cố định nếu prompt template có phần này

Có thể giữ:

1. token số
2. token của entity chính
3. token quan hệ
4. token câu hỏi cuối

### 11.4. Không căn lề token thô giữa các variant

Các variant cùng cấu trúc gần như chắc chắn sẽ có số token khác nhau. Vì vậy, PhiBench không mặc định so sánh token-thứ-`k` của `T_original` với token-thứ-`k` của `T_surface`.

Thay vào đó, việc so sánh được neo theo ngữ nghĩa hoặc theo vai trò trong template:

1. `slot-aligned spans`
   span chứa số, toán tử, quan hệ, hoặc đối tượng được gắn nhãn ngay từ bộ sinh dữ liệu
2. `question-end anchor`
   token cuối của câu hỏi trước khi mô hình bắt đầu sinh đầu ra
3. `answer-zone anchor`
   một cửa sổ token gần đoạn trả lời cuối
4. `prompt-summary`
   trung bình hoặc tóm tắt trên toàn bộ content token, chỉ dùng như đối chứng thô

Nguyên tắc:

1. không dùng token-index alignment làm kết quả chính
2. ít nhất một cách neo cục bộ và một cách tổng hợp toàn prompt phải được báo cáo song song
3. nếu hai cách neo cho kết luận trái ngược, coi đó là dấu hiệu metric chưa ổn định

## 12. Vị trí hook trong Llama-3.2-1B

Ba vị trí candidate:

1. attention output
2. MLP output
3. residual stream sau khi cộng xong block

Cho phase đầu, ưu tiên:

1. residual stream cuối block

Lý do:

1. đây là state tích lũy nhất
2. dễ so sánh giữa layers
3. ít tranh cãi hơn việc chọn riêng attention hay MLP

Có thể mở rộng sau sang:

1. pre-attention residual
2. post-attention residual
3. post-MLP residual

## 13. Giảm chiều

### 13.1. Vấn đề

Hidden size của Llama-3.2-1B quá lớn để ước lượng ổn định covariance và mutual information trực tiếp.

### 13.2. Nguyên tắc

Projector phải được fit một lần trên tập calibration rồi freeze. Không được `fit PCA` riêng cho từng prompt hoặc từng condition, vì như vậy các score sẽ sống trong hệ tọa độ khác nhau và mất khả năng so sánh.

Ngoài ra:

1. projector mặc định không được học trực tiếp từ nhãn đúng-sai của task chính
2. nếu dùng projector có giám sát, nó chỉ được xem là ablation phụ, không phải kết quả chính
3. mọi kết luận chính phải sống được qua ít nhất hai loại projector khác nhau

### 13.3. Pipeline khuyến nghị

1. thu hidden states từ một tập calibration đa dạng nhưng tách biệt test
2. gom các residual vectors và metadata của anchor spans
3. fit nhiều projector candidate:
   random orthogonal projection, PCA, whitened/truncated PCA
4. freeze từng projector
5. khi đánh giá, chỉ `transform`, không `fit`
6. projector có giám sát chỉ chạy ở nhánh phụ để kiểm tra độ nhạy của kết luận

### 13.4. Rủi ro mất tín hiệu cấu trúc

PCA giữ các hướng có phương sai lớn nhất. Điều này không đảm bảo các hướng đó là nơi chứa thông tin cấu trúc logic. Trong LLM, phương sai lớn có thể phản ánh:

1. vị trí token
2. phong cách diễn đạt
3. thành phần cú pháp bề mặt
4. các chiều dị thường có biên độ lớn

Vì vậy, một projector tốt không chỉ cần nén được dữ liệu, mà còn phải cho kết luận ổn định qua các variant cùng cấu trúc.

Biện pháp giảm rủi ro:

1. so sánh PCA với random projection cố định
2. thử loại bỏ một số thành phần chính lớn nhất trước khi tính metric
3. báo cáo độ nhạy theo `d_proj`
4. không dùng projector có giám sát làm kết quả chính

### 13.5. Kích thước nén

Khuyến nghị phase đầu:

1. `d_proj = 64`

Ablation cần thử:

1. `d_proj = 32`
2. `d_proj = 64`
3. `d_proj = 128`

## 14. Phân hoạch A/B

### 14.1. Vì sao cần partition

`Phi-proxy` muốn đo phần thông tin chỉ xuất hiện khi hai phần của state được xét cùng nhau.

### 14.2. Các partition candidate

Không nên chỉ dùng một cách chia. Phase đầu nên hỗ trợ:

1. first-half vs second-half
2. even-index vs odd-index
3. random partition với seed cố định
4. head-aligned partition nếu map được head subspace một cách ổn định

### 14.3. Aggregation qua nhiều partition

Với mỗi layer:

1. tính `Phi-proxy` trên nhiều partition
2. báo cáo mean và variance qua partition

Điều này thay cho minimum information partition exact, vốn quá đắt để duyệt.

## 15. Định nghĩa metric chính

### 15.1. Notation

Sau khi chiếu xuống không gian thấp:

1. `z_l(t) in R^d`

Với một partition `p`, tách:

1. `z_l(t) = [A_l(t); B_l(t)]`

### 15.2. Transition-level integration proxy

Metric chính cho layer `l` và partition `p`:

$$
\Phi_{\text{proxy}}^{(l,p)} =
I\left([A_{l+1}, B_{l+1}] ; [A_l, B_l]\right)
- I\left(A_{l+1}; A_l\right)
- I\left(B_{l+1}; B_l\right)
$$

Diễn giải:

1. hạng đầu đo lượng thông tin của toàn state ở layer `l+1` về toàn state ở layer `l`
2. hai hạng sau đo phần tự bảo toàn của từng nửa nếu xét riêng
3. phần dư là mức coupling cần thiết khi xét cả hai cùng nhau

Đây vẫn chỉ là proxy, không phải exact IIT Phi.

### 15.3. Gaussian estimator

Estimator Gaussian được dùng như baseline vì tính được nhanh, ổn định hơn các estimator phi tham số trong không gian nhiều chiều, và dễ lặp lại. Tuy nhiên, đây chỉ là xấp xỉ làm việc, không phải giả định chân lý về hidden state.

Giả sử vector chiếu gần Gaussian, mutual information giữa hai biến Gaussian nhiều chiều `X` và `Y` được tính bằng:

$$
I(X;Y) = \frac{1}{2}\log \frac{\det \Sigma_X \det \Sigma_Y}{\det \Sigma_{XY}}
$$

Trong đó:

1. `Sigma_X` là covariance của `X`
2. `Sigma_Y` là covariance của `Y`
3. `Sigma_XY` là covariance joint của `[X;Y]`

### 15.4. Kiểm tra độ bền của estimator

Không được dựa toàn bộ kết luận vào một estimator duy nhất. Phase đầu cần tối thiểu ba nhánh:

1. `Gaussian + shrinkage covariance`
   nhánh chính
2. `rank-Gaussianized / copula-style Gaussian`
   để giảm độ nhạy với outlier và phân phối lệch
3. `k-NN MI`
   chỉ dùng như sanity check ở chiều rất thấp hoặc trên subset nhỏ, không dùng làm pipeline chính

Tiêu chuẩn thực dụng:

1. nếu mọi estimator cho cùng chiều hướng kết quả, độ tin cậy tăng
2. nếu chỉ một estimator cho hiệu ứng còn các estimator khác không ủng hộ, không được đưa ra kết luận mạnh

### 15.5. Regularization số học

Khi tính log determinant:

1. thêm ridge `lambda I`
2. theo dõi condition number
3. bỏ những layer-condition quá suy biến nếu estimator không ổn định

### 15.6. Signed và clipped versions

Nên lưu cả hai:

1. `phi_proxy_signed`
2. `phi_proxy_clipped = max(0, phi_proxy_signed)`

Lý do:

1. signed version giúp chẩn đoán estimator
2. clipped version dễ diễn giải hơn khi báo cáo

## 16. Định nghĩa stability

Formula cũ kiểu `1 - Var / mean` không đủ bền. Tài liệu này thay bằng một score bị chặn tốt hơn.

### 16.1. Layer-wise normalized dispersion

Cho một triplet tại layer `l`:

$$
d^{(l)} =
\frac{
\operatorname{std}\left(
\Phi_o^{(l)},
\Phi_s^{(l)},
\Phi_i^{(l)}
\right)
}{
\operatorname{mean}\left(
|\Phi_o^{(l)}|,
|\Phi_s^{(l)}|,
|\Phi_i^{(l)}|
\right) + \epsilon
}
$$

### 16.2. Layer-wise stability

$$
\text{PhiStability}^{(l)} = \exp(-d^{(l)})
$$

Tính chất:

1. luôn nằm trong `(0, 1]`
2. ít nổ hơn khi mean nhỏ
3. dễ diễn giải: dispersion càng thấp, stability càng gần 1

### 16.3. Global stability

$$
\text{PhiStability}_{\text{global}} =
\frac{1}{L} \sum_{l=1}^{L} \text{PhiStability}^{(l)}
$$

### 16.4. Profile similarity bổ sung

Ngoài scalar stability, nên tính thêm similarity của cả profile theo layer:

1. cosine similarity giữa vector `Phi_o[1:L]` và `Phi_s[1:L]`
2. cosine similarity giữa `Phi_o[1:L]` và `Phi_i[1:L]`
3. dynamic time warping không khuyến nghị ở phase đầu vì khó diễn giải

## 17. Metric đầu ra

PhiBench không thay accuracy. Nó thêm signal nội tại bên cạnh accuracy.

Output metrics:

1. exact match
2. numeric match nếu bài có số
3. verifier pass
4. confidence proxy:
   logit margin của đáp án cuối nếu trích được

## 18. Ma trận hành vi

Ma trận làm việc của dự án:

| PhiStability | Accuracy | Diễn giải |
|---|---|---|
| cao | cao | structure-consistent success |
| cao | thấp | process có vẻ đúng hướng nhưng execution sai hoặc output head yếu |
| thấp | cao | possible shortcut or memorized success |
| thấp | thấp | failure cả về process lẫn output |

Lưu ý:

1. đây là nhãn heuristic để phân tích
2. không phải chứng minh hình thức

## 19. Baselines bắt buộc

Nếu không có baseline, rất khó biết `Phi-proxy` có thực sự hữu ích hay chỉ là một số phức tạp.

Baselines nên có:

1. hidden norm theo layer
2. activation variance theo layer
3. attention entropy
4. prompt perplexity hoặc average next-token loss
5. simple mutual information không partition
6. representation similarity giữa variants bằng CKA hoặc cosine

Một kết quả tốt cần cho thấy:

1. `Phi-proxy` hoặc `PhiStability` dự báo OOD accuracy tốt hơn baseline
2. hoặc ít nhất bổ sung tín hiệu ngoài baseline

## 20. Quy trình chạy thí nghiệm

### 20.1. Bước 1. Chuẩn bị dataset

1. sinh families
2. sinh triplets
3. chạy verifier để tạo đáp án vàng
4. tách split

### 20.2. Bước 2. Fit projector

1. lấy calibration prompts
2. chạy model
3. thu residual stream
4. fit projector
5. lưu projector ra disk

### 20.3. Bước 3. Thu hidden states cho eval

1. tokenize prompt
2. forward pass
3. hook residual stream ở mọi layer
4. áp content mask
5. project xuống `d_proj`

### 20.4. Bước 4. Tính metric

1. với mỗi layer
2. với mỗi partition
3. tính `Phi-proxy`
4. aggregate qua token
5. aggregate qua partition

### 20.5. Bước 5. Tính stability

1. gom 3 variants cùng instance
2. tính layer-wise stability
3. tính global stability

### 20.6. Bước 6. Chạy phân tích thống kê

1. tương quan với accuracy
2. so sánh trong và ngoài phân phối
3. bootstrap CI
4. permutation test

## 21. Aggregation qua token và sample

Đây là chỗ dễ sai.

Không nên xem từng token là mẫu độc lập hoàn toàn. Việc gộp token chỉ là một xấp xỉ thực dụng để tạo phân phối mẫu.

Các lựa chọn hợp lý:

1. token-pooled:
   gộp tất cả content token của một prompt thành sample pool
2. anchor-span:
   chỉ lấy các span được gắn nhãn vai trò trong template
3. question-end:
   lấy một cửa sổ nhỏ quanh token cuối của câu hỏi
4. answer-focused:
   chỉ lấy các token gần đoạn trả lời cuối
5. prompt-summary:
   average hidden states theo token rồi mới tính metric

Khuyến nghị phase đầu:

1. chạy ít nhất một chế độ cục bộ:
   `anchor-span` hoặc `question-end`
2. chạy ít nhất một chế độ toàn cục:
   `token-pooled` hoặc `prompt-summary`
3. xem kết quả có nhất quán không
4. bootstrap hoặc permutation phải lấy đơn vị resample là `prompt` hoặc `triplet`, không phải token riêng lẻ

## 22. Kiểm định thống kê

### 22.1. H1

`PhiStability` tương quan dương với OOD accuracy.

Thực hiện:

1. Pearson
2. Spearman
3. bootstrap 95% CI

### 22.2. H2

Tasks nhiều bước hơn có profile sâu hơn.

Thực hiện:

1. so sánh area-under-profile ở nửa sâu và nửa nông
2. Wilcoxon signed-rank giữa early layers và late layers

### 22.3. H3

Trong các trường hợp accuracy cao nhưng OOD shift mạnh, `PhiStability` sẽ tách được success do structure và success do shortcut.

Thực hiện:

1. chia sample thành các nhóm theo đúng/sai
2. so sánh distribution của `PhiStability`
3. tính AUROC cho việc dự báo OOD success

### 22.4. Mô hình hồi quy khuyến nghị

Phase hai có thể dùng:

1. logistic regression hoặc mixed-effects model

Biến phụ thuộc:

1. `correct_on_surface`
2. `correct_on_inverse`

Biến độc lập:

1. `PhiStability_global`
2. `mean_phi_proxy`
3. `prompt_length`
4. `family_type`
5. `difficulty`

## 23. Ablation bắt buộc

Một project kiểu này sẽ không đứng nếu thiếu ablation.

Ít nhất cần có:

1. projector dimension: 32, 64, 128
2. partition type: half, even-odd, random
3. hook location: residual only, residual + attention, residual + MLP
4. token aggregation: anchor-span, question-end, pooled, prompt-summary
5. estimator family: Gaussian, copula-style Gaussian, low-d k-NN sanity check
6. projector family: random projection, PCA, whitened/truncated PCA
7. estimator regularization strength
8. instruction format:
   zero-shot vs short chain-of-thought prompt

## 24. Failure modes cần theo dõi

### 24.1. Estimator artifact

Metric tăng chỉ vì covariance estimation kém ổn định ở chiều cao.

Dấu hiệu:

1. logdet cực đoan
2. condition number rất xấu
3. score đổi mạnh khi tăng ridge rất nhỏ

### 24.2. Prompt length confound

Prompt dài hơn có thể tự làm profile khác dù structure giống nhau.

Cần kiểm soát:

1. length matching
2. regression control cho token count

### 24.3. Lexical novelty confound

OOD prompt lạ từ vựng có thể làm hidden state lệch chỉ vì distributional novelty, không phải vì reasoning demand.

Cách xử lý:

1. có control set "surface novel but logically trivial"
2. so sánh với baseline representation shift

### 24.4. Output-format confound

Mô hình có thể hiểu nhưng fail do format.

Cách xử lý:

1. verifier tolerant
2. normalize numeric extraction

### 24.5. Projector dominance

Kết luận có thể đến từ lựa chọn projector thay vì đến từ cấu trúc bài toán.

Cách xử lý:

1. báo cáo ít nhất hai projector
2. xem hiệu ứng có cùng chiều hay không
3. hạ mức kết luận nếu chỉ một projector cho tín hiệu

### 24.6. Tokenization and anchor confound

Khác biệt về token hóa giữa các variant có thể tạo ra khác biệt profile ngay cả khi cấu trúc logic không đổi.

Cách xử lý:

1. dùng anchor theo span và vai trò ngữ nghĩa
2. không dùng raw token alignment làm mặc định
3. length-match trong phạm vi hợp lý
4. có control set "surface changed but structurally trivial"

## 25. Kết quả kỳ vọng hợp lý

Một pattern hợp lý nếu ý tưởng đúng:

1. Group A:
   accuracy gần nhau, `PhiStability` cao
2. Group B:
   độ sâu suy luận tăng thì profile dịch sang lớp sâu
3. Group C:
   wording tương tự nhưng structure khác thì profile tách nhau rõ
4. Group D:
   với mẫu trả lời đúng, `PhiStability` vẫn tương đối cao;
   với mẫu trả lời sai do không tổng quát hóa, `PhiStability` sụt mạnh

## 26. Tiêu chí thành công phase đầu

Phase đầu được xem là thành công nếu đạt đồng thời:

1. pipeline hook và metric chạy ổn định trên toàn bộ `Llama-3.2-1B`
2. estimator không suy biến trên đa số layers
3. `PhiStability` có tương quan dương với OOD accuracy
4. metric cho tín hiệu bổ sung so với ít nhất một baseline đơn giản
5. kết quả nhất quán tương đối qua nhiều partition
6. kết quả giữ cùng chiều qua ít nhất hai projector
7. kết quả giữ cùng chiều qua ít nhất một anchor cục bộ và một aggregation toàn cục
8. positive control và negative control cho phản ứng đúng chiều

## 27. Cấu trúc code khuyến nghị

```text
phi-bench/
  docs/
    phibench_internal_spec.md
  data/
    raw/
    processed/
    calibration/
  configs/
    model/
    dataset/
    experiment/
  src/
    data_generation/
    prompting/
    hooks/
    projection/
    metrics/
    evaluation/
    analysis/
  outputs/
    activations/
    metrics/
    figures/
```

## 28. Pseudocode tham chiếu

```python
for triplet in benchmark:
    variant_outputs = {}

    for variant_name, prompt in triplet.items():
        tokens = tokenizer(prompt, return_tensors="pt")
        hidden_by_layer = run_model_with_hooks(model, tokens)
        masked = apply_content_mask(hidden_by_layer, tokens)
        projected = projector.transform(masked)

        layer_scores = []
        for layer_idx in range(num_layers - 1):
            z_l = projected[layer_idx]
            z_lp1 = projected[layer_idx + 1]

            partition_scores = []
            for partition in partitions:
                A_l, B_l = split_by_partition(z_l, partition)
                A_lp1, B_lp1 = split_by_partition(z_lp1, partition)
                phi = phi_proxy_gaussian(A_l, B_l, A_lp1, B_lp1)
                partition_scores.append(phi)

            layer_scores.append(mean(partition_scores))

        variant_outputs[variant_name] = layer_scores

    stability = compute_triplet_stability(
        variant_outputs["original"],
        variant_outputs["surface"],
        variant_outputs["inverse"],
    )
```

## 29. Những quyết định thiết kế đang mở

Các câu hỏi chưa khóa:

1. nên lấy residual stream tại mọi token hay chỉ token gần vùng trả lời
2. nên aggregate theo prompt trước hay theo token pool
3. projector tốt nhất là PCA hay random projection
4. có nên thêm causal intervention như activation patching ở phase hai hay không
5. có nên dùng model nhỏ hơn làm smoke test trước khi chạy full `Llama-3.2-1B`

## 30. Roadmap triển khai

### Phase 0. Sanity check

1. tải model
2. hook hidden states
3. lưu activations
4. fit projector
5. tính được metric trên vài prompt toy

### Phase 1. Minimal benchmark

1. chọn 2 groups
2. mỗi group 2 families
3. mỗi family 50 triplets
4. chạy end-to-end

Đầu ra:

1. bảng metric
2. đồ thị profile theo layer
3. báo cáo lỗi estimator

### Phase 2. Full pilot

1. đủ 4 groups
2. đủ baseline
3. đủ ablation chính

### Phase 3. Tightening

1. chọn metric tốt nhất
2. giảm overclaim
3. viết technical report nội bộ

## 31. Tóm tắt định hướng

PhiBench đáng làm nếu được đóng khung đúng:

1. không quảng bá như "đo consciousness"
2. không nói exact IIT Phi khi chưa tính exact IIT
3. tập trung vào một câu hỏi thực nghiệm hẹp nhưng mạnh:
   liệu một integration proxy nội tại có dự báo khả năng tổng quát hóa và giúp phát hiện shortcut tốt hơn accuracy đơn thuần hay không

Nếu giữ được kỷ luật này, dự án có giá trị thật:

1. như một benchmark cơ chế nội tại
2. như một công cụ phát hiện "đúng vì lý do sai"
3. như một lớp trung gian giữa output evaluation và mechanistic interpretability
