function [sample] = proposal_pdf_unif_sample(x, alpha)
    sample = x-alpha + (2*alpha)*rand(1,1);
end