# Generated by Django 4.1 on 2023-10-08 14:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Bitcoin',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(db_index=True)),
                ('open', models.FloatField(null=True)),
                ('high', models.FloatField(db_index=True, null=True)),
                ('low', models.FloatField(db_index=True, null=True)),
                ('close', models.FloatField(db_index=True, null=True)),
                ('volume', models.FloatField(null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Cryptocurrency',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('base_display_symbol', models.CharField(max_length=255, unique=True)),
                ('quote_display_symbol', models.CharField(max_length=255, unique=True)),
                ('product_id', models.CharField(max_length=10, unique=True)),
                ('trading_indicator', models.FloatField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='Ethereum',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(db_index=True)),
                ('open', models.FloatField(null=True)),
                ('high', models.FloatField(db_index=True, null=True)),
                ('low', models.FloatField(db_index=True, null=True)),
                ('close', models.FloatField(db_index=True, null=True)),
                ('volume', models.FloatField(null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddIndex(
            model_name='ethereum',
            index=models.Index(fields=['timestamp', 'high'], name='coinbase_ap_timesta_43d884_idx'),
        ),
        migrations.AddIndex(
            model_name='ethereum',
            index=models.Index(fields=['timestamp', 'low'], name='coinbase_ap_timesta_cb596f_idx'),
        ),
        migrations.AddIndex(
            model_name='ethereum',
            index=models.Index(fields=['timestamp', 'close'], name='coinbase_ap_timesta_6a0a85_idx'),
        ),
        migrations.AddIndex(
            model_name='bitcoin',
            index=models.Index(fields=['timestamp', 'high'], name='coinbase_ap_timesta_1df1d5_idx'),
        ),
        migrations.AddIndex(
            model_name='bitcoin',
            index=models.Index(fields=['timestamp', 'low'], name='coinbase_ap_timesta_ae3a38_idx'),
        ),
        migrations.AddIndex(
            model_name='bitcoin',
            index=models.Index(fields=['timestamp', 'close'], name='coinbase_ap_timesta_9b60a3_idx'),
        ),
    ]