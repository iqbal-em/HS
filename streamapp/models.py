from django.db import models


class DataCam(models.Model):
    id_iklan = models.IntegerField()
    kode_umur = models.CharField(max_length=30)
    kode_gender = models.CharField(max_length=30)
    waktu = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'data_cam'


class Iklan(models.Model):
    nama = models.TextField()
    hitung = models.IntegerField()
    url = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'iklan'
