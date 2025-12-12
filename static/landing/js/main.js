// Main JavaScript functionality for BrandInsight

class BrandInsightApp {
    constructor() {
        this.isMobileMenuOpen = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupScrollAnimations();
        this.setupCounterAnimations();
        this.setupMobileMenu();
        this.setupSolutionsTabs();
        this.setupSmoothScrolling();
    }

    setupEventListeners() {
        // Window load event
        window.addEventListener('load', () => {
            this.handlePageLoad();
        });

        // Scroll event for navbar
        window.addEventListener('scroll', () => {
            this.handleScroll();
        });

        // Resize event
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (this.isMobileMenuOpen && !e.target.closest('.nav-container')) {
                this.closeMobileMenu();
            }
        });
    }

    setupScrollAnimations() {
        // Simple fade-in animation for elements
        const fadeElements = document.querySelectorAll('.feature-card, .testimonial-card, .solution-panel');
        
        const fadeObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, { threshold: 0.1 });

        fadeElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            fadeObserver.observe(el);
        });
    }

    setupCounterAnimations() {
        // Counter animation function
        this.animateCounters = (container) => {
            const counters = container.querySelectorAll('.stat-number');
            
            counters.forEach(counter => {
                const target = parseInt(counter.getAttribute('data-count'));
                const duration = 2000;
                const step = target / (duration / 16);
                let current = 0;
                
                const timer = setInterval(() => {
                    current += step;
                    if (current >= target) {
                        current = target;
                        clearInterval(timer);
                    }
                    counter.textContent = Math.floor(current);
                }, 16);
            });
        };

        // Animate hero stats when they come into view
        const statsObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateCounters(entry.target);
                    statsObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        const heroStats = document.querySelector('.hero-stats');
        if (heroStats) {
            statsObserver.observe(heroStats);
        }
    }

    setupMobileMenu() {
        const mobileToggle = document.getElementById('mobileToggle');
        const navMenu = document.getElementById('navMenu');
        const navActions = document.getElementById('navActions');

        if (mobileToggle) {
            mobileToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleMobileMenu();
            });

            // Close mobile menu when clicking on a link
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', () => {
                    this.closeMobileMenu();
                });
            });
        }
    }

    toggleMobileMenu() {
        const navMenu = document.getElementById('navMenu');
        const navActions = document.getElementById('navActions');
        const mobileToggle = document.getElementById('mobileToggle');

        if (this.isMobileMenuOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }

    openMobileMenu() {
        const navMenu = document.getElementById('navMenu');
        const navActions = document.getElementById('navActions');
        const mobileToggle = document.getElementById('mobileToggle');

        navMenu.classList.add('mobile-open');
        navActions.classList.add('mobile-open');
        mobileToggle.innerHTML = '<i class="fas fa-times"></i>';
        this.isMobileMenuOpen = true;
    }

    closeMobileMenu() {
        const navMenu = document.getElementById('navMenu');
        const navActions = document.getElementById('navActions');
        const mobileToggle = document.getElementById('mobileToggle');

        navMenu.classList.remove('mobile-open');
        navActions.classList.remove('mobile-open');
        mobileToggle.innerHTML = '<i class="fas fa-bars"></i>';
        this.isMobileMenuOpen = false;
    }

    setupSolutionsTabs() {
        const tabButtons = document.querySelectorAll('.solution-tab');
        const tabPanels = document.querySelectorAll('.solution-panel');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');
                
                // Remove active class from all buttons and panels
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanels.forEach(panel => panel.classList.remove('active'));
                
                // Add active class to clicked button and corresponding panel
                button.classList.add('active');
                const targetPanel = document.getElementById(targetTab);
                if (targetPanel) {
                    targetPanel.classList.add('active');
                }
            });
        });
    }

    setupSmoothScrolling() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                
                const targetId = anchor.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    const offsetTop = targetElement.offsetTop - 100;
                    
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });

                    // Close mobile menu if open
                    this.closeMobileMenu();
                }
            });
        });
    }

    handlePageLoad() {
        // Add loaded class to body for transition effects
        document.body.classList.add('loaded');
        
        // Animate hero section
        const heroContent = document.querySelector('.hero-content');
        if (heroContent) {
            heroContent.style.opacity = '0';
            heroContent.style.transform = 'translateY(30px)';
            heroContent.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
            
            setTimeout(() => {
                heroContent.style.opacity = '1';
                heroContent.style.transform = 'translateY(0)';
            }, 300);
        }
    }

    handleScroll() {
        const navbar = document.querySelector('.navbar');
        const scrollY = window.scrollY;
        
        if (scrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }

    handleResize() {
        // Reset mobile menu on larger screens
        if (window.innerWidth > 768 && this.isMobileMenuOpen) {
            this.closeMobileMenu();
        }
    }
}


// FAQ Accordion
document.addEventListener('DOMContentLoaded', function() {
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        
        question.addEventListener('click', () => {
            // Close all other items
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
            
            // Toggle current item
            item.classList.toggle('active');
        });
    });
    
    // Pricing toggle functionality
    const billingToggle = document.getElementById('billingToggle');
    const monthlyPrices = {
        starter: 99,
        professional: 299,
        enterprise: 'Custom'
    };
    const annualPrices = {
        starter: 79,
        professional: 239,
        enterprise: 'Custom'
    };
    
    if (billingToggle) {
        billingToggle.addEventListener('change', function() {
            const isAnnual = this.checked;
            const pricingCards = document.querySelectorAll('.pricing-card');
            
            pricingCards.forEach(card => {
                const plan = card.querySelector('.pricing-header h3').textContent.toLowerCase();
                const priceElement = card.querySelector('.price .amount');
                const periodElement = card.querySelector('.period');
                
                if (plan.includes('starter')) {
                    priceElement.textContent = isAnnual ? annualPrices.starter : monthlyPrices.starter;
                } else if (plan.includes('professional')) {
                    priceElement.textContent = isAnnual ? annualPrices.professional : monthlyPrices.professional;
                } else if (plan.includes('enterprise')) {
                    priceElement.textContent = 'Custom';
                }
                
                if (periodElement) {
                    periodElement.textContent = isAnnual ? '/year' : '/month';
                }
            });
        });
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.brandInsightApp = new BrandInsightApp();
});