// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const createMobileMenu = () => {
        const nav = document.querySelector('nav');
        const navLinks = document.querySelector('.nav-links');
        
        if (nav && navLinks && !document.querySelector('.mobile-menu-btn')) {
            const mobileMenuBtn = document.createElement('div');
            mobileMenuBtn.classList.add('mobile-menu-btn');
            mobileMenuBtn.innerHTML = '☰';
            
            mobileMenuBtn.addEventListener('click', () => {
                navLinks.classList.toggle('active');
                mobileMenuBtn.innerHTML = navLinks.classList.contains('active') ? '✕' : '☰';
            });
            
            nav.appendChild(mobileMenuBtn);
        }
    };
    
    // Initialize mobile menu
    createMobileMenu();
    
    // Handle file upload preview
    const fileInput = document.getElementById('plant_image');
    if (fileInput) {
        const imagePreview = document.getElementById('image-preview');
        const fileName = document.getElementById('file-name');
        
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.addEventListener('load', function() {
                    imagePreview.setAttribute('src', this.result);
                    imagePreview.style.display = 'block';
                    fileName.textContent = file.name;
                });
                
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        const dropArea = document.querySelector('.file-upload-label');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            fileInput.files = files;
            
            // Trigger change event
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    }
    
    // Testimonial slider functionality
    const testimonialSlider = document.querySelector('.testimonials-slider');
    if (testimonialSlider) {
        const testimonials = testimonialSlider.querySelectorAll('.testimonial');
        let currentIndex = 0;
        
        // Only set up auto-sliding if there's more than one testimonial
        if (testimonials.length > 1) {
            setInterval(() => {
                currentIndex = (currentIndex + 1) % testimonials.length;
                testimonialSlider.scrollTo({
                    left: testimonials[currentIndex].offsetLeft,
                    behavior: 'smooth'
                });
            }, 5000); // Change testimonial every 5 seconds
        }
    }
    
    
    // Form validation for crop recommendation
    const cropForm = document.getElementById('crop-recommendation-form');
    if (cropForm) {
        cropForm.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = cropForm.querySelectorAll('input[type="number"]');
            
            inputs.forEach(input => {
                if (!input.value) {
                    isValid = false;
                    input.classList.add('error');
                } else {
                    input.classList.remove('error');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
});