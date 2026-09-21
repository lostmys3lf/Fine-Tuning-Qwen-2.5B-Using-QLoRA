# Fine-tuning Qwen2.5-0.5B-Instruct menggunakan QLoRA

Fine-tuning model untuk format respons layanan pelanggan GadaiKita, menggunakan kuantisasi 4-bit NF4 dan adapter LoRA. Notebook mencakup training, perbandingan output sebelum/sesudah, eksperimen konfigurasi, serta penyimpanan dan pemuatan ulang adapter.

Notebook: [Finetuning_Qwen2.5-0.5B-Instruct_Qlora.ipynb](Finetuning_Qwen2.5-0.5B-Instruct_Qlora.ipynb).

## Hasil eksekusi Google Colab

Hasil berikut diambil dari output `Finetuning_Qwen2_5_0_5B_Instruct_Qlora.ipynb` yang ditinjau pada **21 September 2026**. Kode pada ekspor tersebut sama dengan notebook proyek. Kelima eksperimen menyelesaikan **30 optimizer steps**; evaluasi, grafik, reload adapter utama, dan pembuatan ZIP juga selesai tanpa exception yang tercatat pada output notebook.

Runtime yang tercatat: **Tesla T4, VRAM 14.56 GiB, FP16**, PyTorch `2.11.0+cu128`, Transformers `4.56.2`, PEFT `0.17.1`, bitsandbytes `0.47.0`, TRL `0.23.1`, datasets `4.1.1`, dan Accelerate `1.10.1`.

| Model / rank LoRA | Contoh training | Parameter terlatih | Loss training rata-rata | Waktu training (detik) | Peak VRAM allocated (GiB) | Format sebelum → sesudah |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B / r=16 | 60 | 8,798,208 | 0.6069 | 155.27 | 1.357 | 0% → 100% |
| Qwen2.5-0.5B / r=4 | 60 | 2,199,552 | 1.2555 | 156.59 | 1.807 | 0% → 100% |
| Qwen2.5-0.5B / r=64 | 60 | 35,192,832 | 0.3914 | 151.48 | 2.627 | 0% → 100% |
| Qwen2.5-0.5B / r=16 | 10 | 8,798,208 | 0.7207 | 102.99 | 2.862 | 0% → 100% |
| Qwen2.5-1.5B / r=16 | 60 | 18,464,768 | 0.5445 | 181.55 | 4.474 | 0% → 100% |

Waktu di atas hanya mencakup `trainer.train()`, tidak termasuk download, pemuatan model, generation, atau ekspor. Loss pada tabel merupakan rata-rata training yang dilaporkan trainer. Pada run utama, loss yang dicatat di step 1 sebesar **2.7815** dan di step 30 sebesar **0.0176**.

Peak VRAM merupakan `torch.cuda.max_memory_allocated()` saat training dalam satu session berurutan, bukan total penggunaan GPU atau ukuran adapter. Run rank 4 dan dataset 10 contoh justru mencatat peak lebih tinggi daripada run utama; penyebab selisih belum diisolasi. Perbandingan kebutuhan memori tiap konfigurasi perlu diulang dalam runtime terpisah sebelum menarik kesimpulan tentang pengaruh rank atau ukuran dataset.

## Makna hasil dan kualitas jawaban

**Semua konfigurasi mengikuti salam dan penutup pada 10 prompt parafrasa, tetapi isi jawaban belum konsisten benar.** Skor 100% hanya memeriksa apakah jawaban diawali salam dan diakhiri penutup yang ditentukan; jawaban kosong di antara keduanya juga bisa lolos.

Contoh yang tercatat:

- **Jawaban sesuai dataset:** model utama menjelaskan perpanjangan dengan membayar biaya pemeliharaan di cabang atau melalui aplikasi sebelum jatuh tempo.
- **Format benar, isi keliru:** saat ditanya dokumen gadai emas, model utama tidak menyebut KTP asli dan justru menyebut "pemeliharaan perhiasan".
- **Rank 4 dapat melewatkan isi jawaban:** untuk pertanyaan dokumen, output hanya "Halo, terima kasih telah menghubungi GadaiKita. Ada lagi yang bisa kami bantu?".
- **Rank 64 mencatat loss rata-rata terendah**, tetapi masih mencampur informasi; pertanyaan harga emas dijawab dengan biaya pinjaman "1 persen per 15 hari".
- **Model 1.5B menjawab beberapa topik dengan sesuai**, termasuk jenis barang dan penanganan keterlambatan, tetapi masih salah pada topik lain. Pada pertanyaan pencairan, model menghasilkan informasi jam kerja yang tidak ada di dataset.
- **Pertanyaan di luar dataset memicu informasi tanpa dasar:** model utama menyebut kantor beroperasi 24 jam, padahal jam operasional tidak tersedia dalam data training.

Hasil ini menunjukkan adaptasi format, sementara ketepatan informasi masih perlu diperbaiki. Belum ada skor akurasi isi, validation loss, atau evaluasi lintas seed. Dataset 10 contoh yang juga mendapat 100% tidak membuktikan kualitasnya setara dengan dataset 60 contoh. Evaluasi isi dengan jawaban referensi dan contoh training yang lebih beragam diperlukan sebelum memilih konfigurasi terbaik.

## Konfigurasi dan dataset

- Dataset sintetis GadaiKita berisi 10 pasangan tanya-jawab dengan 6 variasi pertanyaan, total 60 contoh. Informasi perusahaan dan layanan bersifat fiktif.
- Subset 10 contoh mencakup satu contoh per topik. Evaluasi menggunakan 10 parafrasa yang tidak identik dengan prompt training, tetapi topiknya tetap sama; ini belum menguji generalisasi ke topik baru.
- Dua prompt tambahan di luar dataset diperiksa manual dan tidak masuk skor format pada tabel.
- Setiap eksperimen memuat base model baru, memakai seed 42, learning rate `2e-4`, 30 optimizer steps, microbatch 1, dan gradient accumulation 8. Dataset kecil diulang lebih sering.
- Kuantisasi memakai NF4, double quantization, dan optimizer `paged_adamw_8bit`; LoRA memakai alpha `2 x rank` serta dropout `0.05`.
- Target LoRA: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, dan `down_proj`. Loss dihitung pada completion, dengan panjang maksimum 512 token.

## Jalankan di Google Colab

1. Upload notebook dan pilih **Runtime > Change runtime type > T4 GPU**.
2. Gunakan session baru, lalu jalankan sel berurutan. Jika library sudah terimpor sebelum instalasi, restart session setelah instalasi.
3. Seluruh eksperimen aktif secara default. Matikan flag `RUN_RANK_EXPERIMENTS`, `RUN_SMALL_DATASET`, atau `RUN_LARGER_MODEL` jika ingin melewati kelompok tertentu.
4. Simpan notebook beserta output melalui **File > Download > Download .ipynb**. Sel terakhir mengunduh ZIP adapter dan hasil eksperimen.

Notebook proyek disimpan tanpa output; tabel di atas berasal dari ekspor hasil Colab yang diperiksa. Di runtime tersebut, sel instalasi juga mencatat konflik dependensi dengan `diffusers`, `gradio`, dan `gcsfs`. Pipeline fine-tuning tetap selesai, tetapi kompatibilitas dengan penggunaan ketiga library tersebut belum diuji.

## Adapter dan hasil yang disimpan

Setiap run menyimpan adapter beserta tokenizer, `predictions.csv`, `training_log.csv`, `training_metrics.json`, dan `metrics.json`. Hasil gabungan berupa `experiment_metrics.csv`, grafik perbandingan, serta ZIP `finetune-qwen-qlora-results.zip`.

Pada pengujian reload adapter utama, output untuk prompt pertama **identik sebelum disimpan dan setelah dimuat ulang** (`Output identik: True`). Ini memverifikasi satu prompt pada run tersebut, bukan seluruh kemungkinan output.

## Pemeriksaan kode

Notebook memakai deteksi BF16 native dan autocast pada generation sebelum/sesudah training serta saat reload. Pada Tesla T4 yang diuji, dtype yang terpilih adalah FP16. Pemeriksaan inference dilakukan sebelum training; adapter, log, dan metrik training disimpan sebelum evaluasi.

- `tests/test_notebook.py`: struktur notebook, pemilihan dtype, serta regresi benturan activation FP16/BF16 dengan bobot FP32.
- `tests/smoke_training.py`: training dua steps, evaluasi, penyimpanan, dan reload model Qwen2 kecil berbobot acak dengan NF4, untuk rank 4/16/64 dan FP16/BF16. Pengujian lokal ini memakai CPU dan optimizer `adamw_torch`.

Jalankan `python -m unittest discover -s tests -v` untuk regresi, dan `python tests/smoke_training.py` dengan library notebook terpasang untuk pengujian integrasi. Notebook asli tersimpan di `backups/Handson_QLoRA_Finetuning.ipynb`.

## Referensi

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [TRL SFTTrainer 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PyTorch autocast](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch native BF16 detection](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
