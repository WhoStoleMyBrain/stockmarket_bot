# Generated by Django 4.1 on 2023-10-08 14:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coinbase_api', '0002_alter_cryptocurrency_product_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cryptocurrency',
            name='base_display_symbol',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='cryptocurrency',
            name='quote_display_symbol',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterUniqueTogether(
            name='cryptocurrency',
            unique_together={('base_display_symbol', 'quote_display_symbol')},
        ),
    ]