# PhiBench (Working Title): Benchmark đánh giá xử lý cấu trúc trong Transformer bằng tín hiệu nội tại ổn định qua các biến thể đối xứng

Status: working paper draft

Last updated: 2026-03-24

Note: tên benchmark trong tài liệu này chỉ là tên làm việc. Trọng tâm của bản thảo là câu hỏi nghiên cứu và thiết kế thực nghiệm, không phải branding.

## Tóm tắt

Các benchmark hiện nay chủ yếu chấm đúng sai trên đầu ra. Cách chấm này không đủ để phân biệt hai trường hợp rất khác nhau: mô hình trả lời đúng vì dùng cấu trúc logic của bài toán, hay mô hình trả lời đúng vì khớp với mẫu bề mặt quen thuộc trong dữ liệu huấn luyện. Bản thảo này đề xuất `PhiBench`, một benchmark nội tại nhằm đo xem profile xử lý bên trong Transformer có giữ ổn định khi bề mặt bài toán thay đổi nhưng cấu trúc logic giữ nguyên hay không.

Khác với các phát biểu mạnh về ý thức hay suy luận theo nghĩa triết học, chúng tôi chỉ đưa ra một mục tiêu thực dụng hơn: xây dựng một `integration proxy` đo từ hidden states của mô hình, rồi kiểm tra xem chỉ số này có dự báo được khả năng tổng quát hóa ngoài phân phối và có giúp phát hiện các trường hợp "đúng vì lý do sai" tốt hơn accuracy đơn thuần hay không. Benchmark được thiết kế quanh các bộ ba bài toán đối xứng gồm bản gốc, bản đổi bề mặt và bản đảo chiều logic. Chỉ số nội tại được định nghĩa trên chuyển tiếp giữa các lớp của residual stream sau khi nén chiều, có nhiều bộ ước lượng và nhiều cách neo token/span để tránh phụ thuộc vào một lựa chọn kỹ thuật duy nhất.

Đóng góp chính của bản thảo không phải là tuyên bố đã đo được exact `Phi` theo Integrated Information Theory, mà là đề xuất một khung đánh giá chặt chẽ hơn cho câu hỏi sau: khi mô hình thành công, thành công đó có bền theo cấu trúc hay chỉ bám vào bề mặt.

## 1. Giới thiệu

Một mô hình ngôn ngữ có thể đạt điểm cao trên benchmark vì ít nhất hai lý do:

1. mô hình dùng được cấu trúc của bài toán và tổng quát hóa được khi cách diễn đạt thay đổi
2. mô hình nhận ra mẫu bề mặt đã thấy trước đó và kích hoạt một đáp án quen thuộc

Nếu chỉ nhìn output cuối, hai trường hợp này thường bị trộn lẫn. Đây là điểm mù lớn của nhiều benchmark hiện đại. Khi mô hình ngày càng lớn, khả năng nuốt rất nhiều mẫu cũng tăng lên, làm cho benchmark đầu ra càng dễ bị bão hòa mà không cho biết cơ chế thành công bên trong.

Mục tiêu của `PhiBench` là thêm một tầng đánh giá mới. Thay vì chỉ hỏi:

> mô hình có đúng không?

benchmark này hỏi thêm:

> mô hình đúng theo cách nào, và thành công đó có giữ được khi bề mặt của bài toán bị thay đổi hay không?

## 2. Phạm vi và giới hạn của bài thảo

Bản thảo này không cố chứng minh:

1. mô hình có ý thức
2. Transformer có exact `Phi` theo Integrated Information Theory nguyên bản
3. mô hình suy luận thật theo nghĩa triết học mạnh

Thay vào đó, bản thảo chỉ nhắm tới một claim hẹp hơn và kiểm định được hơn:

> tồn tại một chỉ số nội tại đo từ hidden states của Transformer, đủ ổn định và đủ hữu ích để phân biệt thành công bền theo cấu trúc với thành công dựa trên bề mặt

Vì vậy, toàn bộ cách diễn đạt trong bài dùng `Phi-proxy` hoặc `integration proxy`, không dùng `Phi` theo nghĩa hình thức mạnh.

## 3. Câu hỏi nghiên cứu

PhiBench nhắm tới ba câu hỏi:

1. Có thể xây được các bộ bài toán khác bề mặt nhưng cùng cấu trúc logic hay không?
2. Có thể đo được một profile nội tại của mô hình đủ ổn định qua các biến thể cùng cấu trúc hay không?
3. Profile đó có liên hệ với khả năng tổng quát hóa ngoài phân phối và có giúp phát hiện các ca đúng-vì-khớp-mẫu hay không?

## 4. Trực giác cốt lõi

Trực giác trung tâm của dự án là:

1. Nếu mô hình xử lý theo cấu trúc, profile nội tại của nó phải tương đối bất biến khi thay đổi từ vựng, ngữ cảnh và bề mặt câu hỏi nhưng giữ logic.
2. Nếu mô hình chỉ bám bề mặt, profile đó sẽ đổi mạnh khi wording đổi dù cấu trúc toán học hoặc logic giữ nguyên.

Điểm mạnh của trực giác này là nó không buộc phải giải quyết các câu hỏi siêu hình lớn. Nó chỉ đòi một kiểm định thực nghiệm: cùng cấu trúc, khác bề mặt, cơ chế bên trong có còn ổn định không?

## 5. Thiết kế benchmark

### 5.1. Bộ ba đối xứng

Mỗi instance được sinh thành một triplet:

1. `T_original`
   phiên bản gần với dạng bài phổ biến
2. `T_surface`
   cùng cấu trúc nhưng đổi ngữ cảnh, từ vựng, và texture bề mặt
3. `T_inverse`
   cùng lõi quan hệ nhưng đảo chiều câu hỏi hoặc đầu vào/đầu ra

Mục tiêu của triplet là tách:

1. cái thuộc về cấu trúc logic
2. cái thuộc về bề mặt ngôn ngữ

### 5.2. Các nhóm bài toán

Phiên bản pilot nên bắt đầu với bốn nhóm:

1. chuyển bối cảnh
2. kéo dài chuỗi suy luận
3. bẫy ngôn ngữ
4. tri thức giả tưởng ngoài phân phối

Mỗi nhóm chỉ nên dùng các bài có đáp án kiểm được bằng luật, để tránh mơ hồ ở nhãn đúng-sai.

### 5.3. Tiêu chuẩn của một family hợp lệ

Một family chỉ được đưa vào benchmark nếu:

1. cấu trúc logic của nó mô tả được bằng template rõ
2. có thể sinh ít nhất ba biến thể giữ nguyên logic
3. có bộ kiểm đáp án tự động
4. có thể length-match ở mức tương đối
5. có thể gắn nhãn span quan trọng để neo so sánh nội tại

## 6. Định nghĩa trạng thái trong Transformer

### 6.1. Trạng thái

Với prompt đã mã hóa, gọi:

1. `h_l(t)` là residual stream của token `t` tại lớp `l`

Ở đây:

1. `l` là chỉ số lớp
2. `t` là vị trí token

### 6.2. Chuyển tiếp

Ta không đo theo thời gian thật. Ta đo theo độ sâu của mạng:

1. từ `h_l(t)` sang `h_{l+1}(t)`

Đây là một xấp xỉ có chủ đích: layer depth được xem như trục biến đổi thông tin.

### 6.3. Vì sao chọn residual stream

Residual stream là lựa chọn mặc định vì:

1. nó tích lũy thông tin qua attention và MLP
2. nó dễ so sánh giữa các lớp
3. nó ít phụ thuộc vào lựa chọn thành phần hẹp như chỉ riêng attention head

## 7. Chỉ số nội tại

### 7.1. Ý tưởng

Sau khi chiếu trạng thái xuống không gian thấp, ta chia vector thành hai phần:

1. `A_l`
2. `B_l`

Nếu mô hình thật sự có liên kết giữa hai phần này, thì thông tin của toàn hệ ở lớp sau về toàn hệ ở lớp trước phải lớn hơn tổng phần tự bảo toàn của từng nửa khi xét riêng.

### 7.2. Integration proxy

Với một partition `p`, chỉ số tại lớp `l` được định nghĩa:

$$
\Phi_{\text{proxy}}^{(l,p)} =
I\left([A_{l+1}, B_{l+1}] ; [A_l, B_l]\right)
- I\left(A_{l+1}; A_l\right)
- I\left(B_{l+1}; B_l\right)
$$

Đây là một `proxy` cho tính liên kết không tách được giữa hai phần của state. Nó không được diễn giải như exact IIT Phi.

### 7.3. PhiStability

Cho một triplet tại lớp `l`, gọi ba giá trị lần lượt là:

1. `Phi_o^(l)`
2. `Phi_s^(l)`
3. `Phi_i^(l)`

Thay vì dùng công thức dễ nổ kiểu `1 - Var / mean`, ta dùng dispersion chuẩn hóa:

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

và định nghĩa:

$$
\text{PhiStability}^{(l)} = \exp(-d^{(l)})
$$

Chỉ số tổng hợp:

$$
\text{PhiStability}_{\text{global}} =
\frac{1}{L}\sum_{l=1}^{L} \text{PhiStability}^{(l)}
$$

## 8. Ước lượng thực dụng thay vì ước lượng tuyệt đối

### 8.1. Vì sao không đo exact Phi

Exact IIT `Phi` đòi hỏi:

1. hệ phần tử được định nghĩa chặt
2. quan hệ nhân quả được mô tả rõ
3. minimum information partition được duyệt hoặc xấp xỉ nghiêm ngặt

Điều này không khả thi cho Transformer cỡ lớn trong giai đoạn đầu.

### 8.2. Vì sao vẫn dùng được proxy

Mục tiêu của benchmark không phải chứng minh một định lý về ý thức. Mục tiêu là đo một tín hiệu nội tại có ích cho việc phân biệt thành công bền theo cấu trúc với thành công do bám mẫu. Một proxy là chấp nhận được nếu:

1. nó được định nghĩa rõ
2. nó có đối chứng
3. nó ổn định qua các lựa chọn kỹ thuật hợp lý
4. nó có giá trị dự báo vượt qua baseline đơn giản

## 9. Những điểm nghẽn phương pháp phải xử lý

### 9.1. Giả định Gaussian

Hidden states của Transformer không nhất thiết gần chuẩn. Chúng có thể:

1. lệch mạnh
2. có chiều dị thường
3. có đuôi dày

Vì vậy, estimator Gaussian chỉ là baseline.

Thiết kế chặt hơn yêu cầu thêm:

1. `Gaussian + shrinkage covariance`
2. `rank-Gaussianized / copula-style Gaussian`
3. `k-NN MI` chỉ như kiểm tra phụ ở chiều thấp

Kết luận chính chỉ được giữ nếu chiều hướng kết quả nhất quán qua nhiều estimator.

### 9.2. Neo token và span

Các variant cùng cấu trúc không có cùng số token. Vì vậy không so sánh theo kiểu token-thứ-`k` giữa các prompt.

Thay vào đó dùng:

1. `slot-aligned spans`
   span được gắn nhãn ngay từ bộ sinh dữ liệu
2. `question-end anchor`
   token cuối câu hỏi
3. `answer-zone anchor`
   cửa sổ gần đầu ra
4. `prompt-summary`
   chỉ như đối chứng toàn cục

Nếu kết luận chỉ đúng ở một cách neo duy nhất, benchmark chưa đủ chặt.

### 9.3. Rủi ro mất tín hiệu khi nén chiều

PCA có thể giữ lại tín hiệu bề mặt mạnh hơn tín hiệu cấu trúc. Do đó:

1. không dùng một projector duy nhất
2. so sánh random projection với PCA
3. thử bản whitened hoặc loại bỏ vài thành phần chính lớn nhất
4. chỉ dùng projector có giám sát trong ablation phụ

### 9.4. Token không phải mẫu độc lập

Token trong cùng một prompt có phụ thuộc mạnh vào nhau. Vì vậy:

1. không bootstrap theo token
2. chỉ resample theo `prompt` hoặc `triplet`
3. phải báo cáo kết quả cả ở mức cục bộ lẫn mức toàn prompt

## 10. Chương trình kiểm định nền tảng

Trước khi nói tới benchmark hoàn chỉnh, cần qua bốn cổng kiểm định:

### Cổng 1. Dữ liệu đối xứng hợp lệ

Phải chứng minh:

1. triplet cùng cấu trúc thật
2. verifier đúng
3. length và format không lệch quá mạnh

### Cổng 2. Metric có phản ứng đúng trên đối chứng

Cần có:

1. positive control:
   hệ hoặc task dự kiến có liên kết
2. negative control:
   cấu hình tách rời hoặc task bề mặt-trivial

Nếu metric không phản ứng đúng chiều trên hai control này, không nên đi tiếp.

### Cổng 3. Kết quả sống được qua lựa chọn kỹ thuật

Hiệu ứng chính phải giữ cùng chiều qua:

1. ít nhất hai projector
2. ít nhất hai estimator
3. ít nhất một neo cục bộ và một neo toàn cục
4. nhiều partition A/B

### Cổng 4. Có giá trị dự báo

`PhiStability` hoặc chỉ số tương đương phải:

1. tương quan với OOD accuracy
2. hoặc tách được các ca `accuracy cao nhưng nghi là bám mẫu`
3. hoặc bổ sung tín hiệu ngoài baseline đơn giản

## 11. Checkpoint nghiên cứu: phân hoạch, hình thái profile, và overfit-vs-generalization

Phần này ghi lại một checkpoint nhận thức quan trọng của dự án. Mục đích của checkpoint không phải để khẳng định kết luận cuối, mà để chốt những giả thuyết thực nghiệm đủ chặt để dẫn hướng code và thiết kế kiểm định tiếp theo.

### 11.1. Nhát cắt tùy ý và ý nghĩa cấu trúc của phân hoạch

Một phản biện hợp lý là không gian biểu diễn của Transformer là phân tán. Vì vậy, việc chia không gian `2048` chiều thành hai nửa `A/B` không có ý nghĩa ngữ nghĩa tự nhiên. Theo phản biện này, đo lượng thông tin tương hỗ hoặc một `Phi-proxy` trên một nhát cắt như vậy có thể chỉ là một thao tác hình học tùy ý.

Checkpoint hiện tại của dự án chấp nhận điểm phê bình đó ở tầng ngữ nghĩa, nhưng giữ lại một giả thuyết hẹp hơn ở tầng cấu trúc:

> một phân hoạch `A/B` không cần mang ý nghĩa ngữ nghĩa để vẫn có thể mang ý nghĩa như một lát cắt của cấu trúc tính toán

Theo cách nhìn này, thước đo không đại diện cho riêng `A` hay riêng `B`. Nó được dùng như một proxy cho mức độ coupling của trạng thái hệ thống dưới một họ phân hoạch đã chọn. Điều cần kiểm tra không phải là "nhát cắt này có đúng về mặt nghĩa hay không", mà là "kết luận có phụ thuộc quá mạnh vào nhát cắt này hay không".

### 11.2. Độ bền phân hoạch như điều kiện hợp lệ của metric

Từ lập luận trên, một tiêu chuẩn hợp lệ được rút ra:

> nếu `Phi-proxy` thực sự bắt được tín hiệu của tác vụ, thì hiệu ứng do thay đổi task phải lớn hơn hiệu ứng do thay đổi cách cắt

Điều này dẫn tới giả thuyết kiểm định:

1. các phân hoạch hợp lý như `half-half`, `even-odd`, và `random seed` phải cho cùng chiều xu hướng theo lớp
2. độ lệch do thay đổi partition phải nhỏ hơn đáng kể độ lệch do thay đổi họ tác vụ

Viết gọn ở mức khái niệm:

$$
Var(\text{Partition}) \ll Var(\text{Task Family})
$$

Trong bản thảo này, mệnh đề trên không được hiểu như một đẳng thức toán học chính xác, mà như một tiêu chuẩn thiết kế thí nghiệm. Một metric không đạt được điều kiện này thì chưa đủ tư cách đóng vai trò chỉ số trung tâm của benchmark.

### 11.3. Hình thái profile theo lớp là tín hiệu thực nghiệm, không phải tiên đề

Một giả thuyết hấp dẫn là các bài toán đòi hỏi xử lý nhiều bước sẽ tạo ra profile `Phi-proxy` có cấu trúc khác các tác vụ bề mặt. Tuy nhiên, checkpoint hiện tại chủ động tránh ép buộc một hình thái duy nhất như "ngọn núi" hay "một đỉnh ở giữa mạng".

Thay vào đó, bản thảo chốt một kỳ vọng yếu hơn nhưng kiểm định được hơn:

1. tác vụ nhiều thao tác trung gian có thể tạo ra một hoặc nhiều cực đại cục bộ ở các lớp giữa hoặc sâu
2. profile đó phải khác có cấu trúc so với một `null baseline`
3. `null baseline` có thể là chuỗi vô nghĩa giữ độ dài, tác vụ nhận diện bề mặt nông, hoặc bài tra cứu một bước

Điểm quan trọng ở đây là: dự án không đặt trước topology rồi bắt dữ liệu phải khớp. Dữ liệu phải tự quyết định profile là spike, plateau, multi-peak hay dạng khác. Điều duy nhất được yêu cầu là sự thay đổi có cấu trúc phải khác profile nền một cách ổn định.

### 11.4. Overfit-vs-generalization như phép thử giá trị mạnh nhất

Checkpoint này coi thí nghiệm `overfit vs generalization` là phép thử giá trị mạnh nhất của benchmark.

Trong môi trường được kiểm soát, mô hình được huấn luyện vẹt trên tập `A`, rồi so sánh hai trạng thái:

1. bài cũ thuộc đúng tập đã overfit
2. bài mới cùng họ nhưng buộc phải khái quát hóa

Checkpoint hiện tại bác bỏ kỳ vọng quá mạnh rằng bài overfit sẽ cho `Phi-proxy ≈ 0`. Trong Transformer, ngay cả routing theo trí nhớ vẫn đi qua attention, residual, MLP và normalization. Vì vậy, dự án chỉ giữ lại một kỳ vọng thực tế hơn:

1. bài cũ overfit có thể cho profile nông hơn, biên độ thấp hơn, hoặc tập trung trong phạm vi hẹp hơn
2. bài mới cần khái quát có thể cho profile rộng hơn, sâu hơn, hoặc tăng mạnh hơn ở các lớp biến đổi trung gian
3. điều cần chứng minh là chênh lệch thống kê ổn định giữa hai trạng thái, không phải một hình dạng lý tưởng tuyệt đối

Nếu checkpoint này được xác nhận bằng thực nghiệm, benchmark sẽ có một trụ kiểm định rất mạnh: nó không chỉ đo stability giữa các biến thể prompt, mà còn đo được sự khác nhau giữa hai chế độ xử lý thông tin trong một môi trường kiểm soát.

### 11.5. Hệ quả cho roadmap thực nghiệm

Từ checkpoint trên, ba ưu tiên thực nghiệm được rút ra:

1. `partition robustness` phải trở thành tiêu chuẩn pass/fail bắt buộc
2. `task-topology sensitivity` phải được kiểm tra bằng `null baseline` rõ ràng, không chỉ bằng trực giác nhìn đồ thị
3. `overfit-vs-generalization separation` là thí nghiệm giá trị cao nhất sau khi metric cơ bản đã qua được các cổng validity ban đầu

## 12. Thiết kế thí nghiệm đầu tiên

### 12.1. Mô hình

Chọn `Llama-3.2-1B` vì:

1. đủ nhỏ để hook toàn bộ lớp
2. dễ chạy lặp trên phần cứng vừa phải
3. đủ hiện đại để cho kết quả có ý nghĩa

### 12.2. Bộ dữ liệu pilot

Bản pilot chỉ nên có:

1. 2 nhóm bài
2. mỗi nhóm 2 family
3. mỗi family 50 triplet

Mục tiêu pilot không phải chứng minh benchmark thành công, mà là kiểm tra xem tín hiệu có tồn tại và có ổn định ở mức tối thiểu hay không.

### 12.3. Các đầu ra cần có

Pilot phải sinh được:

1. accuracy theo từng variant
2. profile `Phi-proxy` theo lớp
3. `PhiStability` theo layer và toàn cục
4. biểu đồ so sánh giữa projector
5. biểu đồ so sánh giữa anchor
6. log các failure case

## 13. Giả thuyết thực nghiệm

### H1

Triplet cùng cấu trúc sẽ có profile nội tại gần nhau hơn so với bài khác cấu trúc.

### H2

`PhiStability` cao sẽ đi cùng khả năng làm đúng tốt hơn trên `T_surface` và `T_inverse`.

### H3

Bài nhiều bước hơn sẽ có profile dịch sâu hơn trong mạng.

### H4

Một phần đáng kể các trường hợp `accuracy cao nhưng stability thấp` sẽ là các ca thành công không bền, tức đúng vì lý do đáng nghi.

## 14. Cách phân tích kết quả

Phân tích không nên dừng ở một con số trung bình. Cần tối thiểu:

1. profile theo lớp
2. phân bố theo family
3. tách đúng/sai
4. so với baseline đơn giản
5. kiểm tra độ nhạy với lựa chọn kỹ thuật

Một benchmark như thế này chỉ có giá trị khi phần kết luận sống được qua nhiều lát cắt khác nhau.

## 15. Những điều thành công sẽ có nghĩa gì

Nếu dự án thành công, điều đó không có nghĩa:

1. mô hình có ý thức
2. mô hình suy luận như người
3. một số duy nhất có thể đo "độ thông minh"

Điều nó có nghĩa là:

1. ta có một công cụ để phân biệt tốt hơn giữa thành công bền theo cấu trúc và thành công dựa vào bề mặt
2. ta có thể phát hiện các ca "đúng vì lý do sai"
3. ta có thể đánh giá model không chỉ bằng output cuối mà còn bằng độ đáng tin của cơ chế thành công

## 16. Hạn chế

### 16.1. Vấn đề diễn giải

Ngay cả khi `Phi-proxy` hữu ích, nó vẫn chỉ là proxy. Không được diễn giải quá xa những gì dữ liệu hỗ trợ.

### 16.2. Vấn đề estimator

Các estimator khác nhau có thể cho độ lớn khác nhau. Điều quan trọng là chiều hướng và tính ổn định, không phải giá trị tuyệt đối.

### 16.3. Vấn đề projector

Không có đảm bảo rằng không gian thấp đã giữ đúng phần thông tin quan trọng nhất. Vì vậy, projector luôn phải đi kèm ablation.

### 16.4. Vấn đề phân hoạch

Chỉ số có thể nhạy mạnh với cách chia `A/B`. Do đó, mọi kết luận phải báo cáo variance qua nhiều partition.

## 17. Kết luận

Benchmark thế hệ hiện tại thường trả lời tốt câu hỏi:

> mô hình có cho ra đáp án đúng không?

Nhưng chúng trả lời kém hơn câu hỏi:

> khi mô hình đúng, thành công đó có bền theo cấu trúc hay chỉ bám vào dấu hiệu bề mặt?

PhiBench được đề xuất để lấp khoảng trống này. Đóng góp của nó không nằm ở việc khẳng định một lý thuyết lớn về tâm trí hay ý thức, mà ở việc đưa ra một khung đánh giá chặt hơn cho sự khác nhau giữa:

1. đúng vì hiểu cấu trúc
2. đúng nhưng đáng nghi là chỉ khớp mẫu

Nếu khung này đứng được bằng thực nghiệm, nó có thể trở thành một lớp benchmark mới: benchmark không chỉ chấm kết quả, mà còn chấm độ đáng tin của con đường đi tới kết quả đó.
