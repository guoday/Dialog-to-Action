wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
chmod +x gdown.pl
mkdir kb
./gdown.pl https://drive.google.com/open?id=1dgf-Qjvhfv-_EWoDjrTCAY5CwYCw-djt CSQA.zip
./gdown.pl https://drive.google.com/open?id=11DJv1a2Gn2QouJ-9QD-MPTMn3BoCUdAH kb/wikidata_type_dict.json
./gdown.pl https://drive.google.com/open?id=15ctxOZQ68y9cVnZaP-mW9MBpDhRNWG1k kb/wikidata_short_2.json
./gdown.pl https://drive.google.com/open?id=1ST5lqRNlaJlDqZEWe0Nq2Bvl9MyN9vdC kb/wikidata_short_1.json
./gdown.pl https://drive.google.com/open?id=1P-UlZ9vpztw1WWOTPDu8M2rLUA6g3w6q kb/wikidata_rev_type_dict.json
./gdown.pl https://drive.google.com/open?id=15Y57lR3F_5cFdzyyxwsdTXN1qWDiHqOK kb/wikidata_fanout_dict.json
./gdown.pl https://drive.google.com/open?id=1OrAieKryzPlpppEkDUUkbnSkrRSC36y- kb/prop_sub_map5.json
./gdown.pl https://drive.google.com/open?id=1achO9VXrqGRWlUUvt5cdKtSdHcckzkQU kb/prop_obj_map5.json
./gdown.pl https://drive.google.com/open?id=1pzlX_LJjwZFx-wTFzPsi5wIy59QgrQm4 kb/par_child_dict.json
./gdown.pl https://drive.google.com/open?id=1aYEvkZkNRfjyaDqiZz1PO5VAEw5eNphz kb/items_wikidata_n.json
./gdown.pl https://drive.google.com/open?id=1RtkVucx7IGSS46CSwL0rUVrIZgrZ7CW- kb/filtered_property_wikidata4.json
./gdown.pl https://drive.google.com/open?id=1YBGZgK6ultWwZveX3vRr5-MN18TIj39b kb/comp_wikidata_rev.json
./gdown.pl https://drive.google.com/open?id=1rc8iQDUYm6JrUHsghq_MZScChlAWKfzB kb/child_par_dict_save.json
./gdown.pl https://drive.google.com/open?id=1lCw2fFhW6TAzJjmQSNGR4f5EemAxeWa3 kb/child_par_dict_name_2_corr.json
./gdown.pl https://drive.google.com/open?id=1y7df9G8muWV8uHwv78M0rRftmPAGert7 kb/child_par_dict_immed.json
./gdown.pl https://drive.google.com/open?id=1h2NQSyGM-66JU9IYVDz2I-dAavM322Qy kb/child_all_parents_till_5_levels.json
unzip CSQA.zip
mv CSQA_v9 CSQA
rm CSQA.zip
rm gdown.pl
