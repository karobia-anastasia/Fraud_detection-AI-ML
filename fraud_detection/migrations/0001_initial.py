# Generated by Django 5.1.1 on 2024-11-05 15:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Transaction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('step', models.IntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30)], default=1)),
                ('type', models.CharField(choices=[('PAYMENT', 'Payment'), ('TRANSFER', 'Transfer'), ('CASH_OUT', 'Cash Out')], max_length=10)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=12)),
                ('nameOrig', models.CharField(max_length=20)),
                ('oldbalanceOrg', models.DecimalField(decimal_places=2, max_digits=15)),
                ('newbalanceOrig', models.DecimalField(decimal_places=2, max_digits=15)),
                ('nameDest', models.CharField(max_length=20)),
                ('oldbalanceDest', models.DecimalField(decimal_places=2, max_digits=15)),
                ('newbalanceDest', models.DecimalField(decimal_places=2, max_digits=15)),
                ('isFraud', models.BooleanField(default=False)),
                ('isFlaggedFraud', models.BooleanField(default=False)),
            ],
        ),
    ]