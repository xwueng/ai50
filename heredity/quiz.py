# Two factories — Factory A and Factory B — design batteries to be used in
# mobile phones. Factory A produces 60% of all batteries, and Factory B
# produces the other 40%. 2% of Factory A's batteries have defects,
#  and 4% of Factory B's batteries have defects. What is the probability that
#  a battery is both made by Factory A and defective? *
# p(a, d) = p(a) * p(d | a)
# p(a) = 0.6
# p(d | a) = 0.02
pa = 0.6
# p(d | a) = 0.02
p_adefect_rate = 0.02
p_a_and_d = pa * p_adefect_rate
print(f"p(a,d): {p_a_and_d:.4f}")