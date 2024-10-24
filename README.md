# Akbank-Derin-renme-Bootcamp
Balık Sınıflandırma Projesi (Fish Classification Project)
Bu proje, çeşitli balık türlerini görüntülerine dayanarak sınıflandırmak için yapay sinir ağları (ANN) kullanmaktadır. TensorFlow ve Keras kütüphaneleri kullanılarak oluşturulmuş olan bu model, veri artırma teknikleri ve erken durdurma (early stopping) gibi yöntemlerle eğitilmiştir.


# Proje Hakkında
Bu proje, balıkların türlerini sınıflandırmak amacıyla yapay sinir ağı (ANN) kullanarak bir sınıflandırma modeli geliştirmeyi hedeflemektedir. Modelin eğitimi için veri artırma yöntemleri uygulanmış ve sınıflandırma doğruluğunu optimize etmek amacıyla erken durdurma (early stopping) kullanılmıştır.


# Model Özeti
Kullanılan model, Sequential yapısında bir yapay sinir ağıdır.
Modelde 3 yoğun (dense) katman bulunmakta ve her bir katmanda ReLU aktivasyon fonksiyonu kullanılmıştır.
Son katmanda, 9 farklı balık türünü sınıflandırmak için Softmax aktivasyon fonksiyonu kullanılmıştır.


# Sonuçlar
Bu projede, balık sınıflandırma modeli %45.5 doğruluk oranına ulaşmıştır. Eğitim ve doğrulama doğruluğu ile kayıpları şu şekildedir:

Eğitim doğruluğu: 0.31
Doğrulama doğruluğu: 0.35
Test doğruluğu: 0.455
Confusion Matrix ve Sınıflandırma Raporu sonuçları notebook dosyasında bulunmaktadır. Rapor, her sınıf için precision, recall ve f1-score değerlerini içermektedir.



Kaggle Proje linki : https://www.kaggle.com/code/mervekarada/akbank-bootcamp/notebook
